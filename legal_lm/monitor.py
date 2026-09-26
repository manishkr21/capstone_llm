"""Background logger of compute usage: RAM, VRAM, CPU and GPU utilisation.

A daemon thread polls the machine every ``poll_seconds`` (default 5 s) and writes
one CSV row every ``interval_seconds`` (default 60 s). Utilisation columns are the
mean (and max) of the polls in that minute, memory columns are the peak in that
minute, so short spikes are not missed. Rows are appended, so a resumed run keeps
one continuous file; the ``session`` column tells the pieces apart.

    with ResourceMonitor("runs/A1_gpt/resources.csv", device, phase="train") as mon:
        ...
        mon.update(step=step, phase="eval")      # optional: tag rows with progress

Sources: psutil for CPU/RAM, NVML (``nvidia-ml-py``) or ``nvidia-smi`` for the
whole-GPU numbers, and torch's allocator for the memory this process's tensors hold.
Anything unavailable on the current machine (no GPU, no psutil) is left blank.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

try:
    import psutil
except ImportError:  # the monitor still logs torch memory without it
    psutil = None

GB = 1024 ** 3

COLUMNS = [
    "timestamp", "unix_time", "session", "elapsed_min", "window_s", "n_polls", "phase", "step",
    # CPU: system-wide percent (all cores = 100) and this process (1 full core = 100)
    "cpu_pct", "cpu_pct_max", "proc_cpu_pct", "cpu_cores", "load_avg_1m",
    # RAM: whole machine and this process's resident memory
    "ram_used_gb", "ram_total_gb", "ram_pct", "proc_rss_gb",
    # GPU: whole device (NVML / nvidia-smi) ...
    "gpu_name", "gpu_util_pct", "gpu_util_pct_max", "gpu_mem_bw_pct", "vram_used_gb", "vram_total_gb",
    "gpu_power_w", "gpu_temp_c",
    # ... and what PyTorch holds for this process (also filled on Apple MPS)
    "torch_alloc_gb", "torch_reserved_gb", "torch_peak_alloc_gb",
]

MEAN_KEYS = {"cpu_pct", "proc_cpu_pct", "load_avg_1m", "gpu_util_pct", "gpu_mem_bw_pct", "gpu_power_w"}
MAX_KEYS = {"ram_used_gb", "ram_pct", "proc_rss_gb", "vram_used_gb", "gpu_temp_c",
            "torch_alloc_gb", "torch_reserved_gb", "torch_peak_alloc_gb"}
MAX_OF = {"cpu_pct_max": "cpu_pct", "gpu_util_pct_max": "gpu_util_pct"}


# --------------------------------------------------------------------------- GPU readers
class _NvmlReader:
    def __init__(self, index: int):
        import pynvml  # provided by `pip install nvidia-ml-py`
        pynvml.nvmlInit()
        self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(self.h)
        self.name = name.decode() if isinstance(name, bytes) else name

    def read(self) -> dict:
        nv, h = self.nv, self.h
        util, mem = nv.nvmlDeviceGetUtilizationRates(h), nv.nvmlDeviceGetMemoryInfo(h)
        out = {"gpu_util_pct": util.gpu, "gpu_mem_bw_pct": util.memory,
               "vram_used_gb": mem.used / GB, "vram_total_gb": mem.total / GB}
        try:
            out["gpu_power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000
            out["gpu_temp_c"] = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
        except nv.NVMLError:
            pass
        return out


class _SmiReader:
    FIELDS = "name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"

    def __init__(self, index: int):
        self.index = index
        self.name = self._query()[0]

    def _query(self):
        out = subprocess.run(["nvidia-smi", f"--id={self.index}", f"--query-gpu={self.FIELDS}",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return [v.strip() for v in out.stdout.strip().split(",")]

    def read(self) -> dict:
        v = self._query()
        num = lambda s: float(s) if s.replace(".", "", 1).isdigit() else None  # "[N/A]" -> None
        mib = 1024 ** 2 / GB
        return {"gpu_util_pct": num(v[1]), "gpu_mem_bw_pct": num(v[2]),
                "vram_used_gb": num(v[3]) * mib if num(v[3]) is not None else None,
                "vram_total_gb": num(v[4]) * mib if num(v[4]) is not None else None,
                "gpu_power_w": num(v[5]), "gpu_temp_c": num(v[6])}


def _gpu_reader(device: torch.device):
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    for reader in (_NvmlReader, _SmiReader):
        try:
            if reader is _SmiReader and not shutil.which("nvidia-smi"):
                continue
            return reader(index)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- monitor
class ResourceMonitor:
    def __init__(self, csv_path, device="cpu", interval_seconds: float = 60, poll_seconds: float = 5,
                 phase: str = "", enabled: bool = True):
        self.path = Path(csv_path)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.interval, self.poll = float(interval_seconds), min(float(poll_seconds), float(interval_seconds))
        self.enabled = enabled
        self.phase, self.step = phase, None
        self.session = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._polls: list[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self._gpu = _gpu_reader(self.device) if enabled else None
        self._start = time.time()

    # ---- public API
    def update(self, step: Optional[int] = None, phase: Optional[str] = None) -> None:
        """Tag the next rows with the training step and/or the current phase."""
        if step is not None:
            self.step = step
        if phase is not None:
            self.phase = phase

    def start(self) -> "ResourceMonitor":
        if not self.enabled:
            return self
        if psutil is None:
            print("  [monitor] psutil not installed: CPU/RAM columns stay empty (pip install psutil)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if psutil:  # prime the counters: cpu_percent() reports usage since the previous call
            psutil.cpu_percent(None)
            self._proc.cpu_percent(None)
        self._start = self._window_start = time.time()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()
        gpu = self._gpu.name if self._gpu else ("Apple MPS, torch memory only" if self.device.type == "mps" else "none")
        print(f"  [monitor] logging RAM/VRAM/CPU every {self.interval:g}s -> {self.path} (GPU: {gpu})")
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.poll + 15)
        self._thread = None
        try:  # one last poll, so even a stage shorter than poll_seconds leaves a row
            with self._lock:
                self._polls.append(self._sample())
        except Exception:
            pass
        self._flush()  # the last, possibly partial, minute

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- internals
    def _run(self) -> None:
        while not self._stop.wait(self.poll):
            try:
                sample = self._sample()
            except Exception as e:  # never let monitoring crash a training run
                sample = {}
                print(f"  [monitor] sampling failed: {e}")
            with self._lock:
                self._polls.append(sample)
            if time.time() - self._window_start >= self.interval:
                self._flush()

    def _sample(self) -> dict:
        s: dict = {}
        if psutil:
            vm = psutil.virtual_memory()
            s.update(cpu_pct=psutil.cpu_percent(None), proc_cpu_pct=self._proc.cpu_percent(None),
                     ram_used_gb=(vm.total - vm.available) / GB, ram_total_gb=vm.total / GB,
                     ram_pct=vm.percent, proc_rss_gb=self._proc.memory_info().rss / GB,
                     cpu_cores=psutil.cpu_count())
            if hasattr(os, "getloadavg"):
                s["load_avg_1m"] = os.getloadavg()[0]
        if self._gpu is not None:
            try:
                s.update(self._gpu.read())
            except Exception:
                pass
        if self.device.type == "cuda":
            s.update(torch_alloc_gb=torch.cuda.memory_allocated(self.device) / GB,
                     torch_reserved_gb=torch.cuda.memory_reserved(self.device) / GB,
                     torch_peak_alloc_gb=torch.cuda.max_memory_allocated(self.device) / GB)
        elif self.device.type == "mps":
            s.update(torch_alloc_gb=torch.mps.current_allocated_memory() / GB,
                     torch_reserved_gb=torch.mps.driver_allocated_memory() / GB)
        return s

    def _flush(self) -> None:
        with self._lock:
            polls, self._polls = self._polls, []
            window_start, self._window_start = self._window_start, time.time()
        if not polls:
            return
        now = time.time()
        row = {"timestamp": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
               "unix_time": round(now, 1), "session": self.session,
               "elapsed_min": round((now - self._start) / 60, 2), "window_s": round(now - window_start, 1),
               "n_polls": len(polls), "phase": self.phase, "step": self.step,
               "gpu_name": self._gpu.name if self._gpu else ""}
        for key in COLUMNS:
            values = [p[key] for p in polls if p.get(key) is not None]
            if key in MAX_OF:
                values = [p[MAX_OF[key]] for p in polls if p.get(MAX_OF[key]) is not None]
            if not values or key in row:
                continue
            if key in MEAN_KEYS:
                row[key] = sum(values) / len(values)
            elif key in MAX_KEYS or key in MAX_OF:
                row[key] = max(values)
            else:
                row[key] = values[-1]
        row = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def monitor_from_config(cfg: Optional[dict], csv_path, device="cpu", phase: str = "") -> ResourceMonitor:
    """Build a monitor from the ``monitor:`` section of a run config (defaults if absent)."""
    m = (cfg or {}).get("monitor") or {}
    return ResourceMonitor(csv_path, device, interval_seconds=m.get("interval_seconds", 60),
                           poll_seconds=m.get("poll_seconds", 5), phase=phase,
                           enabled=m.get("enabled", True))


# --------------------------------------------------------------------------- reading back
def read_resources(csv_path) -> list[dict]:
    """Load a resources.csv with numbers parsed (empty cells -> None)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k in ("timestamp", "session", "phase", "gpu_name"):
                    continue
                try:
                    r[k] = float(v) if v not in ("", None) else None
                except ValueError:
                    r[k] = None
            rows.append(r)
    return rows


def summarize_resources(csv_path) -> dict:
    """Totals over every session in the file: monitored hours, peaks and average utilisation."""
    rows = read_resources(csv_path) if Path(csv_path).exists() else []
    if not rows:
        return {}

    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]

    def weighted_mean(k):  # weight each minute by how long it actually covered
        pairs = [(r[k], r["window_s"] or 0) for r in rows if r.get(k) is not None]
        total = sum(w for _, w in pairs)
        return round(sum(v * w for v, w in pairs) / total, 2) if total else None

    hours = sum(col("window_s")) / 3600
    gpu_names = sorted({r["gpu_name"] for r in rows if r.get("gpu_name")})
    peak = lambda k: round(max(col(k)), 3) if col(k) else None
    return {
        "monitored_hours": round(hours, 3), "sessions": len({r["session"] for r in rows}),
        "gpu_name": ", ".join(gpu_names) or None, "gpu_hours": round(hours, 3) if gpu_names else None,
        "mean_cpu_pct": weighted_mean("cpu_pct"), "mean_gpu_util_pct": weighted_mean("gpu_util_pct"),
        "peak_ram_gb": peak("ram_used_gb"), "peak_proc_rss_gb": peak("proc_rss_gb"),
        "peak_vram_gb": peak("vram_used_gb"), "peak_torch_alloc_gb": peak("torch_peak_alloc_gb")
        or peak("torch_alloc_gb"),
        "mean_gpu_power_w": weighted_mean("gpu_power_w"),
        "energy_kwh": round(weighted_mean("gpu_power_w") * hours / 1000, 3) if col("gpu_power_w") else None,
    }
