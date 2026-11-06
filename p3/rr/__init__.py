"""Phase 3: Rhetorical Role prediction on top of the Phase-1 legal language models.

Re-uses the Phase-2 package (../p2/lner) for loading the four Phase-1 backbones and for the
LoRA / LoRMA / IA3 / adapter / BitFit / partial implementations; Phase-1 code (legal_lm)
lives in ../capstone_llm.
"""
import sys
from pathlib import Path

P3_ROOT = Path(__file__).resolve().parents[1]
P2_ROOT = (P3_ROOT.parent / "p2").resolve()
P1_ROOT = (P3_ROOT.parent / "capstone_llm").resolve()

for p in (P2_ROOT, P1_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
