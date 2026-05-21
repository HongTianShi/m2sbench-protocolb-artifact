import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def classification_report_dict(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
    }
