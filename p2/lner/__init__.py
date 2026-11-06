"""Phase 2: Legal NER on top of the Phase-1 legal language models.

Phase-1 code (legal_lm) lives in ../capstone_llm; `phase1_path()` puts it on sys.path.
"""
import sys
from pathlib import Path

P2_ROOT = Path(__file__).resolve().parents[1]
P1_ROOT = (P2_ROOT.parent / "capstone_llm").resolve()


def phase1_path() -> Path:
    if str(P1_ROOT) not in sys.path:
        sys.path.insert(0, str(P1_ROOT))
    return P1_ROOT


phase1_path()
