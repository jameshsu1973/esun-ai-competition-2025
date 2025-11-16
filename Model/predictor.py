"""Ensemble 模型預測模組 - 軟投票預測與結果輸出。

此模組負責使用訓練好的 Ensemble 模型進行預測：
- 軟投票：計算三種模型（LightGBM、XGBoost、CatBoost）的平均預測機率
- 閾值轉換：使用最佳閾值將機率轉換為 0/1 標籤
- 結果輸出：儲存競賽提交格式與含機率版本

Classes:
    Predictor: 模型預測器。

Example:
    >>> from Model.predictor import Predictor
    >>> predictor = Predictor(models, scaler, threshold, feature_cols)
    >>> predictions, probs = predictor.predict(df_test)
    >>> predictor.save(predictions, probs, df_test['acct'])
"""

import numpy as np
import pandas as pd
from Config.config import Config

class Predictor:
    """模型預測器 - Ensemble 軟投票預測。
    
    此類別封裝 Ensemble 模型的預測邏輯，包含：
    - 特徵標準化
    - 軟投票：平均 15 個模型（3 類型 × 5 folds）的預測機率
    - 閾值轉換
    - 結果輸出（競賽格式 + 含機率版本）
    
    Attributes:
        models (dict): 模型字典，格式為 {'lgb': [...], 'xgb': [...], 'cat': [...]}u3002
        scaler (StandardScaler): 特徵標準化器。
        threshold (float): 預測閾值。
        feature_cols (list): 特徵欄位名稱清單。
    
    Example:
        >>> predictor = Predictor(trainer.models, trainer.scaler, 
        ...                       trainer.threshold, trainer.feature_cols)
        >>> preds, probs = predictor.predict(df_test)
    """
    
    def __init__(self, models, scaler, threshold, feature_cols):
        """初始化預測器。
        
        Args:
            models (dict): 訓練好的模型字典，格式為 
                {'lgb': [model1, ...], 'xgb': [...], 'cat': [...]}，
                每個 list 包含 5 個 fold 的模型。
            scaler (StandardScaler): 訓練集使用的標準化器。
            threshold (float): 預測閾值，通常為最佳 F1 閾值。
            feature_cols (list): 特徵欄位名稱清單，需與訓練時一致。
        """
        self.models = models
        self.scaler = scaler
        self.threshold = threshold
        self.feature_cols = feature_cols
    
    def predict(self, df_test):
        """預測測試集（Ensemble 軟投票）。
        
        預測流程：
        1. 提取特徵：從 df_test 中提取特徵欄位
        2. 處理缺失值：填充 0
        3. 處理無窮值：替換為 0
        4. 特徵標準化：使用訓練集的 scaler
        5. 軟投票：
           - 對每種模型類型，平均 5 個 fold 的預測機率
           - 平均三種模型的預測機率
        6. 閾值轉換：機率 >= threshold 則為 1，否則為 0
        7. 統計輸出：顯示預測結果與機率分佈
        
        Args:
            df_test (pd.DataFrame): 測試資料，需包含所有特徵欄位。
        
        Returns:
            tuple: 包含兩個元素的元組
                - predictions (np.ndarray): 預測標籤 (0 或 1)
                - probabilities (np.ndarray): 預測機率 (0-1 之間)
        
        Example:
            >>> predictor = Predictor(models, scaler, 0.53, feature_cols)
            >>> predictions, probs = predictor.predict(df_test)
            >>> print(f"預測警示比例: {predictions.mean():.2%}")
            預測警示比例: 4.56%
        """
        print("="*70)
        print("🎯 [模型預測] Ensemble 預測")
        
        X_test = df_test[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_cols)
        
        # 軟投票
        lgb_probs = np.mean([m.predict_proba(X_test_scaled)[:, 1] for m in self.models['lgb']], axis=0)
        xgb_probs = np.mean([m.predict_proba(X_test_scaled)[:, 1] for m in self.models['xgb']], axis=0)
        cat_probs = np.mean([m.predict_proba(X_test_scaled)[:, 1] for m in self.models['cat']], axis=0)
        
        avg_probs = (lgb_probs + xgb_probs + cat_probs) / 3
        predictions = (avg_probs >= self.threshold).astype(int)
        
        print(f"✓ 使用閾值: {self.threshold:.4f}")
        print(f"✓ 預測警示帳戶: {predictions.sum()} ({predictions.mean():.4%})")
        print(f"  機率統計:")
        print(f"    平均: {avg_probs.mean():.4f}")
        print(f"    中位數: {np.median(avg_probs):.4f}")
        print(f"    P90: {np.percentile(avg_probs, 90):.4f}")
        print(f"    P95: {np.percentile(avg_probs, 95):.4f}")
        print(f"    Max: {avg_probs.max():.4f}")
        
        return predictions, avg_probs
    
    def save(self, predictions, probabilities, account_ids, 
             output_path=None):
        """儲存預測結果至 CSV 檔案。
        
        儲存兩個版本：
        1. 競賽提交格式：submission_improved.csv
           - 欄位：acct, label
           - 不包含預測機率
        2. 含機率版本：submission_improved_with_prob.csv
           - 欄位：acct, label, probability
           - 供後續分析使用
        
        Args:
            predictions (np.ndarray): 預測結果 (0/1)。
            probabilities (np.ndarray): 預測機率 (0-1)。
            account_ids (pd.Series or list): 帳戶 ID 清單。
            output_path (str, optional): 輸出檔案路徑。
                若未指定則使用 Config.OUTPUT_PATH + 'submission_improved.csv'。
        
        Side Effects:
            - 建立 output 目錄（若不存在）
            - 儲存 submission_improved.csv
            - 儲存 submission_improved_with_prob.csv
        
        Example:
            >>> predictor.save(predictions, probs, df_test['acct'])
            💾 [儲存結果] 輸出預測結果
            ✓ 已儲存: ./output/submission_improved.csv
            ✓ 含機率版本: ./output/submission_improved_with_prob.csv
        """
        if output_path is None:
            output_path = Config.OUTPUT_PATH + 'submission_improved.csv'
        
        print("="*70)
        print("💾 [儲存結果] 輸出預測結果")
        
        result_df = pd.DataFrame({
            'acct': account_ids,
            'label': predictions,
            'probability': probabilities
        })
        
        # 儲存不含機率版本（競賽提交格式）
        result_df[['acct', 'label']].to_csv(output_path, index=False)
        
        # 儲存含機率版本（供分析使用）
        prob_path = output_path.replace('.csv', '_with_prob.csv')
        result_df.to_csv(prob_path, index=False)
        
        print(f"✓ 已儲存: {output_path}")
        print(f"✓ 含機率版本: {prob_path}")
