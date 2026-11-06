"""Phase 7: prior case retrieval on top of the Phase-1 legal language models.

Re-uses ../p2/lner/peft.py (LoRA, LoRMA, IA3) and ../p3/rr/peft_extra.py (QLoRA NF4, prefix bank,
prompt embedding); ../p4/cj/model.py (chunk encoder); ../p6/summ/model.py for loading the Phase-1 checkpoints.
"""
import sys
from pathlib import Path

P7_ROOT = Path(__file__).resolve().parents[1]
P6_ROOT = (P7_ROOT.parent / "p6").resolve()
P4_ROOT = (P7_ROOT.parent / "p4").resolve()
P3_ROOT = (P7_ROOT.parent / "p3").resolve()
P2_ROOT = (P7_ROOT.parent / "p2").resolve()
P1_ROOT = (P7_ROOT.parent / "capstone_llm").resolve()

for p in (P4_ROOT, P3_ROOT, P2_ROOT, P1_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
