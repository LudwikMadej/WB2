from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score


def get_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'roc_auc':   roc_auc_score(y_true, y_prob),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
    }
