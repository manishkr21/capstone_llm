"""Phase 6: abstractive legal summarization on top of the Phase-1 legal language models.

Re-uses ../p2/lner/peft.py (LoRA, LoRMA, Houlsby adapters) and ../p3/rr/peft_extra.py (QLoRA NF4,
AdaLoRA, prefix bank); the generation wrapper for the four architectures is new (summ/model.py).
"""
import sys
from pathlib import Path

P6_ROOT = Path(__file__).resolve().parents[1]
P3_ROOT = (P6_ROOT.parent / "p3").resolve()
P2_ROOT = (P6_ROOT.parent / "p2").resolve()
P1_ROOT = (P6_ROOT.parent / "capstone_llm").resolve()

for p in (P3_ROOT, P2_ROOT, P1_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
