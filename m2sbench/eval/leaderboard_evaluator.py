import argparse, json
from pathlib import Path
import numpy as np
from .recovery_metrics import evaluate_recovery
from .failure_metrics import evaluate_failure
from .recognition_metrics import classification_report_dict
from .retrieval_metrics import retrieval_metrics

def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser(description="Unified evaluator for M2S-Bench tasks.")
    ap.add_argument("--task", required=True, choices=["recovery","diagnosis","interpretation","retrieval"])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = _load_json(args.predictions)
    out = {"task": args.task, "input_file": args.predictions}
    if args.task == "recovery":
        ref = _load_json(args.reference)
        out.update(evaluate_recovery(np.asarray(pred["points"]), np.asarray(ref["target_points"]), ref.get("budget")))
    elif args.task == "diagnosis":
        ref = _load_json(args.reference)
        rec = evaluate_recovery(np.asarray(pred["points"]), np.asarray(ref["target_points"]), ref.get("budget"))
        out.update(evaluate_failure(np.asarray(pred["points"]), np.asarray(ref["target_points"]), np.asarray(ref.get("source_points")) if ref.get("source_points") is not None else None, rec))
    elif args.task == "interpretation":
        ref = _load_json(args.reference)
        out.update(classification_report_dict(ref["y_true"], pred["y_pred"], ref["labels"]))
    elif args.task == "retrieval":
        ref = _load_json(args.reference)
        out.update(retrieval_metrics(ref["query_labels"], pred["ranked_labels"]))
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
