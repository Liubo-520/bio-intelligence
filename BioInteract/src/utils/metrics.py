"""
metrics.py — Evaluation metrics for DTI prediction.

Includes both standard ML metrics and bioinformatics-specific metrics
like Concordance Index (CI) that are expected in DTI literature.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, mean_squared_error, r2_score
)


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Concordance Index (CI).
    
    CI measures the fraction of concordant pairs: for any two samples,
    if the true affinity of sample i > sample j, the predicted affinity
    of sample i should also be > sample j.
    
    CI is the standard ranking metric in drug-target binding affinity
    prediction (Gönen & Heller, 2005).
    
    Returns value in [0, 1], where 0.5 = random, 1.0 = perfect ranking.
    """
    n = len(y_true)
    if n < 2:
        return 0.5
    
    concordant = 0
    discordant = 0
    tied = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] > y_true[j]:
                if y_pred[i] > y_pred[j]:
                    concordant += 1
                elif y_pred[i] < y_pred[j]:
                    discordant += 1
                else:
                    tied += 1
            elif y_true[i] < y_true[j]:
                if y_pred[i] < y_pred[j]:
                    concordant += 1
                elif y_pred[i] > y_pred[j]:
                    discordant += 1
                else:
                    tied += 1
    
    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    
    return (concordant + 0.5 * tied) / total


def rm2_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the modified r² metric (r²_m).
    
    This metric penalises systematic over/under-prediction and is
    recommended for binding affinity regression (Roy et al., 2009).
    """
    r2 = r2_score(y_true, y_pred)
    
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    
    # correlation coefficient
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    # r²_0 (forced through origin)
    y_pred_scaled = y_pred * (np.sum(y_true * y_pred) / (np.sum(y_pred ** 2) + 1e-8))
    ss_res_0 = np.sum((y_true - y_pred_scaled) ** 2)
    r2_0 = 1 - ss_res_0 / ss_tot
    
    r2_m = r2 * (1 - np.sqrt(abs(r2 - r2_0)))
    
    return float(r2_m)


def classification_metrics(y_true: np.ndarray,
                            y_pred_prob: np.ndarray,
                            threshold: float = None) -> dict:
    """
    Compute all classification metrics for DTI binary prediction.

    If threshold is None, automatically find optimal F1 threshold from
    the precision-recall curve (important for imbalanced datasets).
    """
    if threshold is None:
        # Find optimal threshold from PR curve
        prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_pred_prob)
        f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
        best_idx = np.argmax(f1_arr)
        threshold = float(thresholds[best_idx])

    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    metrics = {
        'AUROC': roc_auc_score(y_true, y_pred_prob),
        'AUPRC': average_precision_score(y_true, y_pred_prob),
        'F1': f1_score(y_true, y_pred_binary),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'threshold': threshold,
    }
    
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all regression metrics for binding affinity prediction.
    """
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'CI': concordance_index(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'r2_m': rm2_index(y_true, y_pred),
        'Pearson': float(np.corrcoef(y_true, y_pred)[0, 1]),
    }
    
    return metrics
