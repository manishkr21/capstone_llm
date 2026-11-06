"""Phase 4: Court Judgment Prediction (binary outcome) on top of the Phase-1 legal language models.

Re-uses ../p2/lner (Phase-1 backbones; full / partial / BitFit / LoRA / LoRMA / adapters) and
../p3/rr/peft_extra.py (AdaLoRA and the other Phase-3 PEFT methods).
"""
import sys
from pathlib import Path

P4_ROOT = Path(__file__).resolve().parents[1]
P3_ROOT = (P4_ROOT.parent / "p3").resolve()
P2_ROOT = (P4_ROOT.parent / "p2").resolve()
P1_ROOT = (P4_ROOT.parent / "capstone_llm").resolve()

for p in (P3_ROOT, P2_ROOT, P1_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
