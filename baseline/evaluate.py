import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def calculate_score(y, y_pred):
    precision_micro = precision_score(y, y_pred, average='micro')
    precision_macro = precision_score(y, y_pred, average='macro')

    recall_micro = recall_score(y, y_pred, average='micro')
    recall_macro = recall_score(y, y_pred, average='macro')

    f1_micro = f1_score(y, y_pred, average='micro')
    f1_macro = f1_score(y, y_pred, average='macro')

    conf_matrix = confusion_matrix(y, y_pred, normalize=None)
    norm_true_cm = np.nan_to_num(
        confusion_matrix(y, y_pred, normalize="true"), 0.)
    norm_pred_cm = np.nan_to_num(
        confusion_matrix(y, y_pred, normalize="pred"), 0.)

    overall_accuracy = accuracy_score(y, y_pred)
    average_accuracy = np.diag(norm_true_cm).mean()
    return {
        "prediction": y_pred.tolist(),
        "accuracy": {"overall": overall_accuracy, "average": average_accuracy},
        "recall": {"micro": recall_micro, "macro": recall_macro},
        "precision": {"micro": precision_micro, "macro": precision_macro},
        "f1": {"micro": f1_micro, "macro": f1_macro},
        "confusion_matrix": {
            "none": conf_matrix.tolist(),
            "true": norm_true_cm.tolist(),
            "pred": norm_pred_cm.tolist()
        }
    }


def evaluate(model, x, y):
    y_pred = model.predict(x)
    return calculate_score(y, y_pred)
