"""Create a tiny synthetic 'legal' dataset in the same folder layout as the real one,
so the whole pipeline can be tested in minutes without downloading gigabytes.

The four files deliberately use different JSON layouts (array of objects, JSON Lines,
id-keyed object, array of strings) to exercise the auto-detecting reader.

    python scripts/make_toy_data.py --out data_toy/hf
"""
import argparse
import json
import random
from pathlib import Path

NAMES = ["Ramesh Kumar", "Sunita Devi", "M/s Bharat Steels Ltd.", "the State of Punjab",
         "Union of India", "Anil Sharma", "the Commissioner of Income Tax", "Meera Iyer"]
COURTS = ["High Court of Delhi", "Supreme Court of India", "Income Tax Appellate Tribunal",
          "High Court of Bombay", "National Green Tribunal"]
ACTS = ["Section 302 of the Indian Penal Code", "Article 226 of the Constitution",
        "Section 138 of the Negotiable Instruments Act", "Section 11 of the Arbitration Act",
        "Order XXXIX Rule 1 of the Code of Civil Procedure", "Section 43B of the Income Tax Act"]
TEMPLATES = [
    "The appellant {a} has preferred this appeal against the judgment of the {c}.",
    "Learned counsel for the respondent {b} submitted that the order is not sustainable under {s}.",
    "It is well settled that the burden of proof lies upon the prosecution under {s}.",
    "The {c} observed that the petitioner {a} had failed to establish a prima facie case.",
    "In view of the above discussion, the appeal is allowed and the impugned order is set aside.",
    "The question that arises for consideration is whether {s} applies to the facts of this case.",
    "Heard learned counsel for the parties and perused the material available on record.",
    "The respondent {b} contended that the notice issued under {s} was barred by limitation.",
    "Accordingly, the writ petition is disposed of with no order as to costs.",
    "We find no merit in the submission advanced on behalf of {a} and the same is rejected.",
]


def document(rng: random.Random) -> str:
    n = rng.randint(80, 140)
    return " ".join(rng.choice(TEMPLATES).format(a=rng.choice(NAMES), b=rng.choice(NAMES),
                                                c=rng.choice(COURTS), s=rng.choice(ACTS))
                    for _ in range(n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data_toy/hf")
    ap.add_argument("--docs", type=int, default=150)
    args = ap.parse_args()
    rng = random.Random(0)
    root = Path(args.out) / "Phase_1_Pretraining"
    (root / "Train").mkdir(parents=True, exist_ok=True)
    (root / "Test").mkdir(parents=True, exist_ok=True)
    docs = {name: [document(rng) for _ in range(args.docs)]
            for name in ("acts", "high_courts", "supreme_court", "tribunals")}

    json.dump([{"id": i, "text": t} for i, t in enumerate(docs["acts"])],
              open(root / "Train/train_acts.json", "w"), indent=1)                    # array of objects
    with open(root / "Train/train_supreme_court.json", "w") as f:                     # JSON Lines
        for i, t in enumerate(docs["supreme_court"]):
            f.write(json.dumps({"id": i, "text": t}) + "\n")
    json.dump({str(i): {"text": t} for i, t in enumerate(docs["high_courts"])},
              open(root / "Train/train_high_courts.json", "w"))                       # id -> object
    json.dump(docs["tribunals"], open(root / "Train/train_tribunals.json", "w"))       # array of strings

    test = {str(i): {"text": " ".join(document(rng).split()[:512])} for i in range(1, 9)}
    json.dump(test, open(root / "Test/test.json", "w"), indent=1)
    print(f"toy dataset written to {root}")
