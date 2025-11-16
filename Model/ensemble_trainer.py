"""Ensemble 模型訓練模組 - LightGBM + XGBoost + CatBoost。

此模組實作 Ensemble Learning 方法，結合三種梯度提升樹模型：
- LightGBM: 基於 Histogram 的快速 GBDT
- XGBoost: 經典梯度提升決策樹
- CatBoost: 專注於類別特徵的 GBDT

訓練流程：
1. 5-Fold Stratified Cross-Validation
2. 每個 fold 訓練三個模型
3. 收集驗證集預測機率
4. 搜索最佳閾值以最大化 F1 Score
5. 輸出特徵重要性與閾值分析圖

Classes:
    EnsembleTrainer: Ensemble 模型訓練器。

Example:
    >>> from Model.ensemble_trainer import EnsembleTrainer
    >>> from Config.config import Config
    >>> trainer = EnsembleTrainer(Config)
    >>> trainer.train(df_train)
    >>> print(f"最佳閾值: {trainer.threshold:.4f}")
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from Config.config import Config
from Utils.evaluation import find_best_threshold
from Utils.visualization import plot_threshold_analysis

class EnsembleTrainer:
    """Ensemble 模型訓練器 - 訓練並管理三種 GBDT 模型。
    
    此類別負責完整的 Ensemble 訓練流程，包含：
    - 特徵標準化
    - 5-Fold Stratified Cross-Validation
    - 訓練 LightGBM、XGBoost、CatBoost
    - 自動閾值搜索
    - 特徵重要性分析
    
    Attributes:
        config: 配置物件（預設為 Config）。
        models (dict): 模型字典，格式為 {'lgb': [...], 'xgb': [...], 'cat': [...]},
            每個 list 包含 5 個 fold 的模型。
        scaler (StandardScaler): 特徵標準化器。
        threshold (float): 最佳預測閾值。
        feature_cols (list): 特徵欄位名稱清單。
        feature_importance (pd.DataFrame): 特徵重要性排序。
    
    Example:
        >>> trainer = EnsembleTrainer()
        >>> trainer.train(df_train)
        >>> print(f"訓練了 {len(trainer.models['lgb'])} 個 LightGBM 模型")
        5
    """
    
    def __init__(self, config=None):
        """初始化 Ensemble 訓練器。
        
        Args:
            config: 配置物件，包含超參數、交叉驗證設定等。
                若未指定則使用 Config 類別。
        """
        self.config = config or Config
        self.models = {'lgb': [], 'xgb': [], 'cat': []}
        self.scaler = None
        self.threshold = 0.5
        self.feature_cols = None
        self.feature_importance = None
    
    def train(self, df_train):
        """訓練 Ensemble 模型並搜索最佳閾值。
        
        完整訓練流程：
        1. 準備特徵：提取特徵欄位，處理缺失值和無窮值
        2. 特徵標準化：使用 StandardScaler
        3. 交叉驗證：5-Fold Stratified CV
        4. 模型訓練：每個 fold 訓練 LightGBM、XGBoost、CatBoost
        5. 閾值搜索：在驗證集上搜索最佳 F1 閾值（0.1-0.9）
        6. 視覺化：繪製閾值分析圖
        7. 特徵重要性：輸出 Top 15 重要特徵
        
        Args:
            df_train (pd.DataFrame): 訓練資料，需包含：
                - acct: 帳戶 ID
                - label: 標籤（1=警示, 0=正常）
                - 62 個特徵欄位
        
        Side Effects:
            - 更新 self.models（15 個模型：3 類型 × 5 folds）
            - 更新 self.scaler（標準化器）
            - 更新 self.threshold（最佳閾值）
            - 更新 self.feature_cols（特徵名稱）
            - 更新 self.feature_importance（特徵重要性）
            - 儲存 threshold_analysis.png（閾值分析圖）
        
        Example:
            >>> trainer = EnsembleTrainer()
            >>> trainer.train(df_train)
            🚀 [模型訓練] Ensemble 模型 (5-Fold CV)
            ...
            🎯 最佳閾值: 0.5300
        """
        print("="*70)
        print(f"🚀 [模型訓練] Ensemble 模型 ({self.config.N_SPLITS}-Fold CV)")
        
        feature_cols = [c for c in df_train.columns if c not in ['acct', 'label']]
        X = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df_train['label']
        
        print(f"特徵數: {len(feature_cols)}, 樣本數: {len(X)}")
        
        # 標準化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # 交叉驗證
        skf = StratifiedKFold(
            n_splits=self.config.N_SPLITS, 
            shuffle=True, 
            random_state=self.config.RANDOM_SEED
        )
        
        all_val_probs = []
        all_val_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            print(f"\n--- Fold {fold+1}/{self.config.N_SPLITS} ---")
            
            X_tr, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            print(f"訓練:{len(X_tr):,} (警示:{y_tr.sum()}), 驗證:{len(X_val):,} (警示:{y_val.sum()})")
            
            # 訓練三個模型
            lgb_model = self._train_lightgbm(X_tr, y_tr, pos_weight)
            xgb_model = self._train_xgboost(X_tr, y_tr, pos_weight)
            cat_model = self._train_catboost(X_tr, y_tr, pos_weight)
            
            self.models['lgb'].append(lgb_model)
            self.models['xgb'].append(xgb_model)
            self.models['cat'].append(cat_model)
            
            # Ensemble 預測
            lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
            xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
            cat_probs = cat_model.predict_proba(X_val)[:, 1]
            avg_probs = (lgb_probs + xgb_probs + cat_probs) / 3
            
            all_val_probs.extend(avg_probs)
            all_val_labels.extend(y_val)
        
        # 尋找最佳閾值
        print("\n" + "="*70)
        print("🔍 尋找最佳閾值...")
        
        all_val_probs = np.array(all_val_probs)
        all_val_labels = np.array(all_val_labels)
        
        best_threshold, best_f1, threshold_results = find_best_threshold(
            all_val_labels, all_val_probs,
            threshold_min=self.config.THRESHOLD_MIN,
            threshold_max=self.config.THRESHOLD_MAX,
            threshold_step=self.config.THRESHOLD_STEP
        )
        
        print(f"\n🎯 最佳閾值: {best_threshold:.4f}")
        print(f"   對應 F1: {best_f1:.4f}")
        print(f"   Precision: {threshold_results[best_threshold]['precision']:.4f}")
        print(f"   Recall: {threshold_results[best_threshold]['recall']:.4f}")
        
        pred_count = np.sum(all_val_probs >= best_threshold)
        print(f"   驗證集預測警示: {pred_count}/{len(all_val_labels)} ({pred_count/len(all_val_labels):.2%})")
        
        self.threshold = best_threshold
        self.feature_cols = feature_cols
        
        # 繪製閾值分析圖
        plot_threshold_analysis(threshold_results, best_threshold)
        
        # 特徵重要性
        self._print_feature_importance(feature_cols)
    
    def _train_lightgbm(self, X, y, pos_weight):
        """訓練 LightGBM 模型。
        
        Args:
            X (pd.DataFrame): 特徵矩陣（已標準化）。
            y (pd.Series): 標籤向量。
            pos_weight (float): 正樣本權重，用於處理類別不平衡。
        
        Returns:
            lgb.LGBMClassifier: 訓練好的 LightGBM 模型。
        """
        params = self.config.LGBM_PARAMS.copy()
        params['scale_pos_weight'] = pos_weight
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        return model
    
    def _train_xgboost(self, X, y, pos_weight):
        """訓練 XGBoost 模型。
        
        Args:
            X (pd.DataFrame): 特徵矩陣（已標準化）。
            y (pd.Series): 標籤向量。
            pos_weight (float): 正樣本權重，用於處理類別不平衡。
        
        Returns:
            xgb.XGBClassifier: 訓練好的 XGBoost 模型。
        """
        params = self.config.XGB_PARAMS.copy()
        params['scale_pos_weight'] = pos_weight
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)
        return model
    
    def _train_catboost(self, X, y, pos_weight):
        """訓練 CatBoost 模型。
        
        Args:
            X (pd.DataFrame): 特徵矩陣（已標準化）。
            y (pd.Series): 標籤向量。
            pos_weight (float): 正樣本權重，用於處理類別不平衡。
        
        Returns:
            CatBoostClassifier: 訓練好的 CatBoost 模型。
        """
        params = self.config.CAT_PARAMS.copy()
        params['scale_pos_weight'] = pos_weight
        model = CatBoostClassifier(**params)
        model.fit(X, y)
        return model
    
    def _print_feature_importance(self, feature_cols):
        """輸出 Top 15 特徵重要性（基於 LightGBM）。
        
        使用第一個 fold 的 LightGBM 模型計算特徵重要性，
        並以 DataFrame 格式儲存至 self.feature_importance。
        
        Args:
            feature_cols (list): 特徵欄位名稱清單。
        
        Side Effects:
            - 更新 self.feature_importance
            - 輸出 Top 15 特徵到 console
        """
        print("\n🏆 Top 15 重要特徵 (LightGBM):")
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.models['lgb'][0].feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in self.feature_importance.head(15).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.1f}")
