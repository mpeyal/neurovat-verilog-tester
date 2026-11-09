"""Classification metrics that count rejections (no output spike) as errors."""
import numpy as np


def confusion(y_true, y_pred, n):
    """n x (n+1): true class rows, predicted class columns, last = rejected."""
    cm = np.zeros((n, n + 1), dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError("Target/prediction length mismatch")
    for t, p in zip(y_true, y_pred):
        if not 0 <= t < n:
            raise ValueError("Target outside class range")
        cm[int(t), int(p) if 0 <= p < n else n] += 1
    return cm


def prf1(cm):
    """Per-class precision/recall/F1 + support and the overall accuracy,
    macro-F1 and support-weighted F1 from a confusion matrix."""
    cm = np.asarray(cm, float)
    tp = np.diag(cm)
    fp = cm[:, :len(tp)].sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / np.maximum(tp + fn, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    support = cm.sum(1)
    tot = max(cm.sum(), 1.0)
    return {
        "prec": prec, "rec": rec, "f1": f1, "support": support.astype(int),
        "acc": float(tp.sum() / tot),
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1.0)),
    }
