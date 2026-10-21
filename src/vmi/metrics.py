"""정답(GT) mask와 예측 mask를 비교하는 지표."""
import numpy as np


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """픽셀 단위 IoU / Precision / Recall.

    precision이 낮다 = 배경을 사람으로 잘못 포함했다 (false positive)
    recall이 낮다    = 사람의 일부를 놓쳤다 (false negative)
    """
    p, g = pred > 0, gt > 0
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }
