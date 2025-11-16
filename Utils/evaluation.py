""" 評估指標計算模組 - F1 Score 最佳閾值搜索。

此模組提供評估指標計算與閾值最佳化功能：
- 搜索使 F1 Score 最大化的閾值
- 同時計算 Precision 和 Recall
- 支援自定義閾值範圍與步長

Functions:
    find_best_threshold: 搜索最佳 F1 閾值。

Example:
    >>> from Utils.evaluation import find_best_threshold
    >>> best_thresh, best_f1, results = find_best_threshold(
    ...     y_true, y_probs, threshold_min=0.1, threshold_max=0.9
    ... )
    >>> print(f"最佳閾值: {best_thresh:.4f}, F1: {best_f1:.4f}")
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def find_best_threshold(y_true, y_probs, threshold_min=0.1, threshold_max=0.9, threshold_step=0.01):
    """尋找最佳閾值以最大化 F1 Score。
    
    此函數在指定範圍內搜索不同閾值，計算每個閾值對應的
    F1 Score、Precision 和 Recall，並返回使 F1 Score 最大的閾值。
    
    搜索方法：
    1. 生成閾值序列：np.arange(threshold_min, threshold_max, threshold_step)
    2. 對每個閾值：
       - 將預測機率轉換為 0/1 標籤
       - 計算 F1、Precision、Recall
    3. 找出 F1 Score 最高的閾值
    
    Args:
        y_true (np.ndarray or list): 真實標籤 (0 或 1)。
        y_probs (np.ndarray or list): 預測機率 (0-1 之間)。
        threshold_min (float, optional): 最小閾值。預設為 0.1。
        threshold_max (float, optional): 最大閾值。預設為 0.9。
        threshold_step (float, optional): 閾值步長。預設為 0.01。
    
    Returns:
        tuple: 包含三個元素的元組
            - best_threshold (float): 最佳閾值
            - best_f1 (float): 最佳 F1 Score
            - results (dict): 所有閾值的結果字典，格式為
                {threshold: {'f1': ..., 'precision': ..., 'recall': ...}}
    
    Example:
        >>> y_true = [0, 0, 1, 1, 0, 1]
        >>> y_probs = [0.2, 0.3, 0.7, 0.8, 0.1, 0.9]
        >>> best_thresh, best_f1, results = find_best_threshold(
        ...     y_true, y_probs, threshold_min=0.3, threshold_max=0.8, threshold_step=0.1
        ... )
        >>> print(f"最佳閾值: {best_thresh:.2f}")
        最佳閾值: 0.50
        >>> print(f"Precision: {results[best_thresh]['precision']:.2f}")
        >>> print(f"Recall: {results[best_thresh]['recall']:.2f}")
    """
    thresholds = np.arange(threshold_min, threshold_max, threshold_step)
    results = {}
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        results[thresh] = {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # 找出最佳閾值
    best_threshold = max(results.keys(), key=lambda t: results[t]['f1'])
    best_f1 = results[best_threshold]['f1']
    
    return best_threshold, best_f1, results
