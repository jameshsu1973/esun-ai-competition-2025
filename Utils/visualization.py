"""視覺化工具模組 - 閾值分析與效能評估圖表。

此模組提供視覺化功能，用於呈現模型評估結果：
- 閾值分析圖：展示 F1、Precision、Recall vs Threshold 曲線
- 標記最佳閾值位置
- 高解析度圖表輸出

Functions:
    plot_threshold_analysis: 繪製閾值分析圖。

Example:
    >>> from Utils.visualization import plot_threshold_analysis
    >>> plot_threshold_analysis(
    ...     threshold_results, best_threshold=0.53, 
    ...     save_path='threshold_analysis.png'
    ... )
    ✓ 閾值分析圖已儲存: threshold_analysis.png
"""

import matplotlib.pyplot as plt

def plot_threshold_analysis(threshold_results, best_threshold, save_path='threshold_analysis.png'):
    """繪製閾值分析圖。
    
    此函數繪製 F1 Score、Precision、Recall 隨閾值變化的曲線圖，
    並標記最佳閾值位置，幫助分析模型在不同閾值下的表現。
    
    圖表元素：
    - F1 Score：藍色實線
    - Precision：綠色虛線
    - Recall：紅色虛線
    - 最佳閾值：橘色垂直虛線 + 紅色圓點
    - 網格線：半透明灰色
    
    Args:
        threshold_results (dict): 閾值結果字典，格式為
            {threshold: {'f1': ..., 'precision': ..., 'recall': ...}}。
            通常由 find_best_threshold() 函數返回。
        best_threshold (float): 最佳閾值，將在圖中標記。
        save_path (str, optional): 儲存路徑。預設為 'threshold_analysis.png'。
    
    Side Effects:
        - 儲存圖表至指定路徑（PNG 格式，300 DPI）
        - 關閉圖表視窗（plt.close()）
    
    Example:
        >>> results = {
        ...     0.3: {'f1': 0.75, 'precision': 0.6, 'recall': 0.9},
        ...     0.5: {'f1': 0.83, 'precision': 0.8, 'recall': 0.85},
        ...     0.7: {'f1': 0.78, 'precision': 0.9, 'recall': 0.7}
        ... }
        >>> plot_threshold_analysis(results, best_threshold=0.5)
        ✓ 閾值分析圖已儲存: threshold_analysis.png
    """
    thresholds = sorted(threshold_results.keys())
    f1_scores = [threshold_results[t]['f1'] for t in thresholds]
    precisions = [threshold_results[t]['precision'] for t in thresholds]
    recalls = [threshold_results[t]['recall'] for t in thresholds]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=2)
    plt.plot(thresholds, precisions, 'g--', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r--', label='Recall', linewidth=2)
    
    # 標記最佳閾值
    best_f1 = threshold_results[best_threshold]['f1']
    plt.axvline(x=best_threshold, color='orange', linestyle=':', linewidth=2, 
               label=f'Best Threshold={best_threshold:.3f}')
    plt.plot(best_threshold, best_f1, 'ro', markersize=10)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Threshold Analysis for Fraud Detection', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 儲存圖片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 閾值分析圖已儲存: {save_path}")
    plt.close()
