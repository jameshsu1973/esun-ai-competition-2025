"""
改進版詐欺偵測系統 - 修正階段順序
特點：
1. 正確的階段執行順序
2. 分離的訓練/測試特徵快取
3. 高活躍帳戶採樣策略
4. Ensemble 模型
5. 詳細評估指標
"""

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== 配置參數 ====================
RANDOM_SEED = 42
TRAIN_FEATURE_CACHE_PATH = "features_train.pkl"
TEST_FEATURE_CACHE_PATH = "features_test.pkl"
MODEL_PATH = "ensemble_models.pkl"

class ImprovedFraudDetector:
    
    def __init__(self, dir_path='./preliminary_data/'):
        self.dir_path = dir_path
        
    # ==================== 資料載入 ====================
    def load_data(self):
        print("="*70)
        print("📂 [階段1] 載入資料集")
        self.df_txn = pd.read_csv(f'{self.dir_path}acct_transaction.csv')
        self.df_alert = pd.read_csv(f'{self.dir_path}acct_alert.csv')
        self.df_predict = pd.read_csv(f'{self.dir_path}acct_predict.csv')
        
        # 預處理時間
        self.df_txn['txn_hour'] = pd.to_datetime(
            self.df_txn['txn_time'], format='%H:%M:%S', errors='coerce'
        ).dt.hour.fillna(12)

        # 幣值轉換
        exchange_rates = {
            'AUD': 19.6, 'CAD': 21.7, 'CHF': 37.74, 'CNY': 4.2,
            'EUR': 35, 'GBP': 40.2, 'HKD': 3.82, 'JPY': 0.2,
            'NZD': 17.5, 'SEK': 3.15, 'SGD': 23.24, 'THB': 0.92,
            'TWD': 1, 'USD': 30, 'ZAR': 1.72
        }
        self.df_txn['txn_amt'] = self.df_txn.apply(
            lambda row: row['txn_amt'] * exchange_rates.get(row['currency_type'], 1), 
            axis=1
        )
        print(f"✓ 交易:{len(self.df_txn):,} | 警示:{len(self.df_alert):,} | 待測:{len(self.df_predict):,}")
    
    # ==================== 特徵工程 ====================
    def extract_features_batch(self, account_list, stage_name="特徵提取"):
        print(f"\n🔧 [{stage_name}] 提取 {len(account_list):,} 個帳戶的特徵...")
        start_time = time.time()
        
        features_list = []
        for i, acct in enumerate(account_list):
            if (i+1) % 1000 == 0:
                elapsed = time.time() - start_time
                speed = (i+1) / elapsed
                remaining = (len(account_list) - i - 1) / speed
                print(f"  進度: {i+1:>6}/{len(account_list)} ({i/len(account_list)*100:>5.1f}%) | "
                      f"速度: {speed:.1f} accts/s | 剩餘: {remaining/60:.1f} min")
            
            out_txns = self.df_txn[self.df_txn['from_acct'] == acct]
            in_txns = self.df_txn[self.df_txn['to_acct'] == acct]
            all_txns = pd.concat([out_txns, in_txns])
            
            features = {'acct': acct}
            
            if len(all_txns) == 0:
                features.update(self._get_empty_features())
            else:
                features.update(self._extract_account_features(out_txns, in_txns, all_txns))
            
            features_list.append(features)
        
        elapsed = time.time() - start_time
        print(f"✅ 特徵提取完成 | 耗時: {elapsed:.1f}s | 速度: {len(account_list)/elapsed:.1f} accts/s")
        return pd.DataFrame(features_list)
    
    def _extract_account_features(self, out_txns, in_txns, all_txns):
        features = {}
        
        # ===== 1. 基本交易統計 =====
        features['out_count'] = len(out_txns)
        features['in_count'] = len(in_txns)
        features['total_count'] = len(all_txns)
        features['txn_ratio'] = features['out_count'] / max(features['in_count'], 1)
        features['in_out_diff'] = abs(features['out_count'] - features['in_count'])
        
        # ===== 2. 金額特徵（統一轉為 TWD） =====
        if len(out_txns) > 0:
            features['out_amt_sum'] = out_txns['txn_amt'].sum()
            features['out_amt_mean'] = out_txns['txn_amt'].mean()
        
        if len(out_txns) > 0:
            features['out_amt_sum'] = out_txns['txn_amt'].sum()
            features['out_amt_mean'] = out_txns['txn_amt'].mean()
            features['out_amt_std'] = out_txns['txn_amt'].std()
            features['out_amt_max'] = out_txns['txn_amt'].max()
            features['out_amt_min'] = out_txns['txn_amt'].min()
            features['out_amt_median'] = out_txns['txn_amt'].median()
            features['out_amt_cv'] = features['out_amt_std'] / (features['out_amt_mean'] + 1e-6)
            features['out_amt_q25'] = out_txns['txn_amt'].quantile(0.25)
            features['out_amt_q75'] = out_txns['txn_amt'].quantile(0.75)
            features['out_amt_iqr'] = features['out_amt_q75'] - features['out_amt_q25']
        else:
            for k in ['sum','mean','std','max','min','median','cv','q25','q75','iqr']:
                features[f'out_amt_{k}'] = 0
        
        if len(in_txns) > 0:
            features['in_amt_sum'] = in_txns['txn_amt'].sum()
            features['in_amt_mean'] = in_txns['txn_amt'].mean()
            features['in_amt_median'] = in_txns['txn_amt'].median()
        else:
            for k in ['sum','mean','median']:
                features[f'in_amt_{k}'] = 0
        
        features['amt_in_out_ratio'] = features['in_amt_sum'] / max(features['out_amt_sum'], 1)
        features['amt_balance'] = features['in_amt_sum'] - features['out_amt_sum']
        
        # ===== 3. 時間特徵 =====
        features['date_range'] = all_txns['txn_date'].max() - all_txns['txn_date'].min() + 1
        features['txn_velocity'] = features['total_count'] / features['date_range']
        features['active_days'] = all_txns['txn_date'].nunique()
        features['active_day_ratio'] = features['active_days'] / features['date_range']
        
        # 夜間/工作時間交易
        features['night_txn_count'] = len(all_txns[(all_txns['txn_hour'] < 6) | (all_txns['txn_hour'] >= 22)])
        features['night_txn_ratio'] = features['night_txn_count'] / features['total_count']
        features['work_hour_count'] = len(all_txns[(all_txns['txn_hour'] >= 9) & (all_txns['txn_hour'] < 18)])
        features['work_hour_ratio'] = features['work_hour_count'] / features['total_count']
        
        # 首尾週交易
        min_date = all_txns['txn_date'].min()
        max_date = all_txns['txn_date'].max()
        first_week = all_txns[all_txns['txn_date'] <= min_date + 7]
        last_week = all_txns[all_txns['txn_date'] >= max_date - 7]
        features['first_week_ratio'] = len(first_week) / features['total_count']
        features['last_week_ratio'] = len(last_week) / features['total_count']
        features['first_last_week_diff'] = abs(features['first_week_ratio'] - features['last_week_ratio'])
        
        # ===== 4. 對手方特徵 =====
        features['unique_from'] = in_txns['from_acct'].nunique()
        features['unique_to'] = out_txns['to_acct'].nunique()
        features['unique_counterparties'] = features['unique_from'] + features['unique_to']
        features['counterparty_diversity'] = features['unique_counterparties'] / max(features['total_count'], 1)
        
        # ===== 5. 交易類型特徵 =====
        features['self_txn_count'] = len(all_txns[all_txns['is_self_txn'] == 'Y'])
        features['self_txn_ratio'] = features['self_txn_count'] / features['total_count']
        
        features['unique_channels'] = all_txns['channel_type'].nunique()
        features['channel_diversity'] = features['unique_channels'] / features['total_count']
        
        features['foreign_currency_count'] = len(all_txns[all_txns['currency_type'] != 'TWD'])
        features['foreign_currency_ratio'] = features['foreign_currency_count'] / features['total_count']
        features['unique_currencies'] = all_txns['currency_type'].nunique()
        
        features['cross_bank_out'] = len(out_txns[out_txns['to_acct_type'] == 2])
        features['cross_bank_in'] = len(in_txns[in_txns['from_acct_type'] == 2])
        features['cross_bank_total'] = features['cross_bank_out'] + features['cross_bank_in']
        features['cross_bank_ratio'] = features['cross_bank_total'] / features['total_count']
   
        # ===== 6. 風險指標 =====
        # 大額交易
        if len(out_txns) > 0:
            threshold_90 = out_txns['txn_amt'].quantile(0.9)
            large_txns = out_txns[out_txns['txn_amt'] > threshold_90]
            features['large_txn_count'] = len(large_txns)
            features['large_txn_ratio'] = features['large_txn_count'] / len(out_txns)
            features['large_txn_amt_sum'] = large_txns['txn_amt'].sum()
            features['large_txn_amt_ratio'] = features['large_txn_amt_sum'] / features['out_amt_sum']
        else:
            for k in ['count','ratio','amt_sum','amt_ratio']:
                features[f'large_txn_{k}'] = 0
        
        # 小額交易
        if len(out_txns) > 0:
            threshold_10 = out_txns['txn_amt'].quantile(0.1)
            features['small_txn_count'] = len(out_txns[out_txns['txn_amt'] < threshold_10])
            features['small_txn_ratio'] = features['small_txn_count'] / len(out_txns)
        else:
            features['small_txn_count'] = 0
            features['small_txn_ratio'] = 0
        
        # 集中度指標
        if features['out_count'] > 0:
            to_counts = out_txns['to_acct'].value_counts()
            features['out_concentration'] = to_counts.iloc[0] / features['out_count']
            features['out_top3_concentration'] = to_counts.head(3).sum() / features['out_count']
        else:
            features['out_concentration'] = 0
            features['out_top3_concentration'] = 0
        
        if features['in_count'] > 0:
            from_counts = in_txns['from_acct'].value_counts()
            features['in_concentration'] = from_counts.iloc[0] / features['in_count']
        else:
            features['in_concentration'] = 0
        
        # 爆發分數
        daily_counts = all_txns.groupby('txn_date').size()
        if len(daily_counts) > 0:
            features['max_daily_txns'] = daily_counts.max()
            features['avg_daily_txns'] = daily_counts.mean()
            features['std_daily_txns'] = daily_counts.std()
            features['burst_score'] = features['max_daily_txns'] / (features['avg_daily_txns'] + 1e-6)
        else:
            for k in ['max_daily_txns','avg_daily_txns','std_daily_txns','burst_score']:
                features[k] = 0
        
        # 時間熵
        hour_dist = all_txns['txn_hour'].value_counts(normalize=True)
        features['hour_entropy'] = -np.sum(hour_dist * np.log(hour_dist + 1e-10))
        
        features['unique_dates'] = all_txns['txn_date'].nunique()
        features['date_coverage'] = features['unique_dates'] / features['date_range']
        
        return features
    
    def _get_empty_features(self):
        return {k: 0 for k in [
            'out_count','in_count','total_count','txn_ratio','in_out_diff',
            'out_amt_sum','out_amt_mean','out_amt_std','out_amt_max','out_amt_min','out_amt_median',
            'out_amt_cv','out_amt_q25','out_amt_q75','out_amt_iqr',
            'in_amt_sum','in_amt_mean','in_amt_median','amt_in_out_ratio','amt_balance',
            'date_range','txn_velocity','active_days','active_day_ratio',
            'night_txn_count','night_txn_ratio','work_hour_count','work_hour_ratio',
            'first_week_ratio','last_week_ratio','first_last_week_diff',
            'unique_from','unique_to','unique_counterparties','counterparty_diversity',
            'self_txn_count','self_txn_ratio','unique_channels','channel_diversity',
            'foreign_currency_count','foreign_currency_ratio','unique_currencies',
            'cross_bank_out','cross_bank_in','cross_bank_total','cross_bank_ratio',
            'large_txn_count','large_txn_ratio','large_txn_amt_sum','large_txn_amt_ratio',
            'small_txn_count','small_txn_ratio','out_concentration','out_top3_concentration',
            'in_concentration','max_daily_txns','avg_daily_txns','std_daily_txns',
            'burst_score','hour_entropy','unique_dates','date_coverage'
        ]}
    
    # ==================== 訓練資料準備 ====================
    def prepare_training_data(self, use_cache=True, sampling_strategy='high_activity_10x'):
        print("="*70)
        print(f"📊 [階段2] 準備訓練資料 - 策略: {sampling_strategy}")
        
        # 檢查快取
        if use_cache and os.path.exists(TRAIN_FEATURE_CACHE_PATH):
            print(f"✓ 載入訓練特徵快取: {TRAIN_FEATURE_CACHE_PATH}")
            cache_data = joblib.load(TRAIN_FEATURE_CACHE_PATH)
            df_features = cache_data['df_train']
            print(f"✓ 特徵數: {len(df_features.columns)-2}, 樣本數: {len(df_features):,}")
            print(f"✓ 警示比例: {df_features['label'].mean():.4%}")
            return df_features
        
        alert_accts = set(self.df_alert['acct'].unique())
        print(f"警示帳戶數: {len(alert_accts)}")
        
        # 計算所有玉山帳戶的交易量
        esun_out = self.df_txn[self.df_txn['from_acct_type'] == 1].groupby('from_acct').size()
        esun_in = self.df_txn[self.df_txn['to_acct_type'] == 1].groupby('to_acct').size()
        txn_counts = pd.concat([esun_out, esun_in]).groupby(level=0).sum().sort_values(ascending=False)
        
        # 排除警示帳戶
        non_alert_txn_counts = txn_counts[~txn_counts.index.isin(alert_accts)]
        
        # 採樣策略
        if sampling_strategy == 'high_activity_10x':
            top_pct = 0.3
            multiplier = 10
        elif sampling_strategy == 'high_activity_5x':
            top_pct = 0.5
            multiplier = 5
        else:  # all
            top_pct = 1.0
            multiplier = 1
        
        top_n = int(len(non_alert_txn_counts) * top_pct)
        high_activity_accts = non_alert_txn_counts.head(top_n).index.tolist()
        
        sample_size = min(len(high_activity_accts), len(alert_accts) * multiplier)
        np.random.seed(RANDOM_SEED)
        sampled_non_alert = np.random.choice(high_activity_accts, size=sample_size, replace=False)
        
        print(f"可選高活躍帳戶: {len(high_activity_accts):,}, 實際採樣: {len(sampled_non_alert):,}")
        
        train_accts = list(alert_accts) + list(sampled_non_alert)
        print(f"訓練帳戶總數: {len(train_accts):,} (警示:{len(alert_accts)} + 採樣:{len(sampled_non_alert)})")
        
        # 提取特徵
        df_features = self.extract_features_batch(train_accts, stage_name="階段2-訓練特徵提取")
        df_features['label'] = df_features['acct'].apply(lambda x: 1 if x in alert_accts else 0)
        
        print(f"✓ 特徵數: {len(df_features.columns)-2}, 警示比例: {df_features['label'].mean():.4%}")
        
        # 快取
        if use_cache:
            joblib.dump({'df_train': df_features}, TRAIN_FEATURE_CACHE_PATH, compress=3)
            print(f"✓ 訓練特徵已快取至 {TRAIN_FEATURE_CACHE_PATH}")
        
        return df_features
    
    def prepare_test_data(self, use_cache=True):
        print("="*70)
        print("🎯 [階段3] 準備測試資料")
        
        # 檢查快取
        if use_cache and os.path.exists(TEST_FEATURE_CACHE_PATH):
            print(f"✓ 載入測試特徵快取: {TEST_FEATURE_CACHE_PATH}")
            df_features = joblib.load(TEST_FEATURE_CACHE_PATH)
            print(f"✓ 特徵數: {len(df_features.columns)-1}, 樣本數: {len(df_features):,}")
            return df_features
        
        test_accts = self.df_predict['acct'].tolist()
        print(f"測試帳戶數: {len(test_accts)}")
        
        # 提取特徵
        df_features = self.extract_features_batch(test_accts, stage_name="階段3-測試特徵提取")
        
        # 快取
        if use_cache:
            joblib.dump(df_features, TEST_FEATURE_CACHE_PATH, compress=3)
            print(f"✓ 測試特徵已快取至 {TEST_FEATURE_CACHE_PATH}")
        
        return df_features
    
    # ==================== 模型訓練 ====================
    def train_ensemble_cv(self, df_train, n_splits=5, threshold=0.5, find_best_threshold=True):
        print("="*70)
        print(f"🚀 [階段4] 訓練 Ensemble 模型 ({n_splits}-Fold CV)")
        
        feature_cols = [c for c in df_train.columns if c not in ['acct','label']]
        X = df_train[feature_cols].fillna(0).replace([np.inf,-np.inf], 0)
        y = df_train['label']
        
        print(f"特徵數: {len(feature_cols)}, 樣本數: {len(X)}")
        
        # 標準化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # 交叉驗證
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        
        self.models = {'lgb':[], 'xgb':[], 'cat':[]}
        cv_metrics = {'f1':[], 'precision':[], 'recall':[]}
        all_val_probs = []
        all_val_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            print(f"\n--- Fold {fold+1}/{n_splits} ---")
            
            X_tr, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            print(f"訓練:{len(X_tr):,} (警示:{y_tr.sum()}), 驗證:{len(X_val):,} (警示:{y_val.sum()}), 權重:{pos_weight:.1f}")
            
            # LightGBM
            lgb_model = lgb.LGBMClassifier(
                objective='binary', num_leaves=31, learning_rate=0.05,
                n_estimators=500, scale_pos_weight=pos_weight,
                random_state=RANDOM_SEED, verbosity=-1
            )
            lgb_model.fit(X_tr, y_tr)
            self.models['lgb'].append(lgb_model)
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(
                max_depth=6, learning_rate=0.05, n_estimators=500,
                scale_pos_weight=pos_weight, random_state=RANDOM_SEED, verbosity=0
            )
            xgb_model.fit(X_tr, y_tr)
            self.models['xgb'].append(xgb_model)
            
            # CatBoost
            cat_model = CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=6,
                scale_pos_weight=pos_weight, random_seed=RANDOM_SEED, verbose=False
            )
            cat_model.fit(X_tr, y_tr)
            self.models['cat'].append(cat_model)
            
            # Ensemble 預測
            lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
            xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
            cat_probs = cat_model.predict_proba(X_val)[:, 1]
            avg_probs = (lgb_probs + xgb_probs + cat_probs) / 3
            
            # 儲存用於尋找最佳閾值
            all_val_probs.extend(avg_probs)
            all_val_labels.extend(y_val)
            
            ensemble_pred = (avg_probs > threshold).astype(int)
            
            f1 = f1_score(y_val, ensemble_pred)
            precision = precision_score(y_val, ensemble_pred, zero_division=0)
            recall = recall_score(y_val, ensemble_pred, zero_division=0)
            
            cv_metrics['f1'].append(f1)
            cv_metrics['precision'].append(precision)
            cv_metrics['recall'].append(recall)
            
            print(f"Ensemble (閾值={threshold}) - F1:{f1:.4f}, Precision:{precision:.4f}, Recall:{recall:.4f}")
            
            # 混淆矩陣
            cm = confusion_matrix(y_val, ensemble_pred)
            tn, fp, fn, tp = cm.ravel()
            print(f"混淆矩陣: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        print("\n" + "="*70)
        print(f"📊 交叉驗證結果 (閾值={threshold}):")
        print(f"  F1:        {np.mean(cv_metrics['f1']):.4f} ± {np.std(cv_metrics['f1']):.4f}")
        print(f"  Precision: {np.mean(cv_metrics['precision']):.4f} ± {np.std(cv_metrics['precision']):.4f}")
        print(f"  Recall:    {np.mean(cv_metrics['recall']):.4f} ± {np.std(cv_metrics['recall']):.4f}")
        
        # 尋找最佳閾值
        if find_best_threshold:
            print("\n" + "="*70)
            print("🔍 尋找最佳閾值...")
            all_val_probs = np.array(all_val_probs)
            all_val_labels = np.array(all_val_labels)
            
            best_threshold, best_f1, threshold_results = self._find_best_threshold(
                all_val_labels, all_val_probs
            )
            
            print(f"\n🎯 最佳閾值: {best_threshold:.4f}")
            print(f"   對應 F1 分數: {best_f1:.4f}")
            print(f"   Precision: {threshold_results[best_threshold]['precision']:.4f}")
            print(f"   Recall: {threshold_results[best_threshold]['recall']:.4f}")
            
            self.threshold = best_threshold
            self.threshold_results = threshold_results
            
            # 繪製閾值分析圖
            self._plot_threshold_analysis(threshold_results)
        else:
            self.threshold = threshold
        
        # 特徵重要性（取第一個 fold 的 LightGBM）
        print("\n🏆 Top 15 重要特徵 (LightGBM):")
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.models['lgb'][0].feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in feature_imp.head(15).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.1f}")
        
        self.feature_cols = feature_cols
        self.feature_importance = feature_imp
        
        return cv_metrics
    
    def _find_best_threshold(self, y_true, y_probs):
        """尋找最佳閾值以最大化 F1 分數"""
        thresholds = np.arange(0.1, 0.9, 0.01)
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
    
    def _plot_threshold_analysis(self, threshold_results):
        """繪製閾值分析圖"""
        thresholds = sorted(threshold_results.keys())
        f1_scores = [threshold_results[t]['f1'] for t in thresholds]
        precisions = [threshold_results[t]['precision'] for t in thresholds]
        recalls = [threshold_results[t]['recall'] for t in thresholds]
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=2)
        plt.plot(thresholds, precisions, 'g--', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, 'r--', label='Recall', linewidth=2)
        
        # 標記最佳閾值
        best_threshold = max(threshold_results.keys(), key=lambda t: threshold_results[t]['f1'])
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
        plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ 閾值分析圖已儲存: threshold_analysis.png")
        plt.close()
    
    # ==================== 預測 ====================
    def predict_ensemble(self, df_test):
        print("="*70)
        print("🎯 [階段5] Ensemble 預測")
        
        X_test = df_test[self.feature_cols].fillna(0).replace([np.inf,-np.inf], 0)
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_cols)
        
        # 軟投票
        lgb_probs = np.mean([m.predict_proba(X_test_scaled)[:, 1] for m in self.models['lgb']], axis=0)
        xgb_probs = np.mean([m.predict_proba(X_test_scaled)[:, 1] for m in self.models['xgb']], axis=0)
        cat_probs = np.mean([m.predict_proba(X_test_scaled)[:, 1] for m in self.models['cat']], axis=0)
        
        avg_probs = (lgb_probs + xgb_probs + cat_probs) / 3
        predictions = (avg_probs > self.threshold).astype(int)
        
        print(f"✓ 預測警示帳戶: {predictions.sum()} ({predictions.mean():.4%})")
        print(f"  平均機率: {avg_probs.mean():.4f}")
        print(f"  機率分布: Min={avg_probs.min():.4f}, Max={avg_probs.max():.4f}, "
              f"Median={np.median(avg_probs):.4f}, P95={np.percentile(avg_probs, 95):.4f}")
        
        return predictions, avg_probs
    
    def save_predictions(self, predictions, probs, output_path='submission_improved.csv'):
        print("="*70)
        print("💾 [階段6] 儲存結果")
        
        result_df = pd.DataFrame({
            'acct': self.df_predict['acct'],
            'label': predictions,
            'probability': probs
        })
        
        result_df[['acct', 'label']].to_csv(output_path, index=False)
        result_df.to_csv(output_path.replace('.csv', '_with_prob.csv'), index=False)
        
        print(f"✓ 已儲存: {output_path}")
        print(f"✓ 含機率版本: {output_path.replace('.csv', '_with_prob.csv')}")

# ==================== 主程式 ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  改進版詐欺偵測系統")
    print("  整合高活躍採樣 + Ensemble + 豐富特徵")
    print("="*70)
    
    detector = ImprovedFraudDetector()
    
    # 階段1: 載入資料
    detector.load_data()
    
    # 階段2: 準備訓練資料
    df_train = detector.prepare_training_data(
        use_cache=True,
        sampling_strategy='high_activity_10x'  # 可選: high_activity_10x, high_activity_5x, all
    )
    
    # 階段3: 準備測試資料
    df_test = detector.prepare_test_data(use_cache=True)
    
    # 階段4: 訓練 Ensemble 模型
    cv_metrics = detector.train_ensemble_cv(
        df_train, 
        n_splits=5, 
        threshold=0.5,  # 初始閾值
        find_best_threshold=True  # 自動尋找最佳閾值
    )
    
    # 階段5: 預測
    predictions, probs = detector.predict_ensemble(df_test)
    
    # 階段6: 儲存結果
    detector.save_predictions(predictions, probs, output_path='result_improved.csv')
    
    print("\n" + "="*70)
    print("✅ 完成！")
    print("="*70)