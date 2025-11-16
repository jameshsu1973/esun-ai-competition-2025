"""特徵工程模組 - 從交易資料中提取 62 維特徵向量。

此模組負責從原始交易資料中提取豐富的特徵，用於詐欺偵測模型訓練。
特徵涵蓋基本統計、金額特徵、時間特徵、對手方特徵、交易類型與風險指標。

Classes:
    FeatureEngineer: 特徵工程器，提供訓練集與測試集特徵提取功能。

Features (62 維):
    - 基本交易統計 (5 個): 交易次數、轉入轉出比例等
    - 金額特徵 (23 個): 總額、均值、標準差、分位數等
    - 時間特徵 (13 個): 日期範圍、活躍天數、夜間交易比例等
    - 對手方特徵 (4 個): 交易對象多樣性、集中度等
    - 交易類型特徵 (11 個): 跨行交易、外幣交易、自轉交易等
    - 風險指標 (14 個): 大額交易比例、集中度、時間熵等

Example:
    >>> from Preprocess.feature_engineering import FeatureEngineer
    >>> engineer = FeatureEngineer(df_txn)
    >>> df_train = engineer.extract_train_features(df_alert, use_cache=True)
    >>> print(f"特徵維度: {len(df_train.columns) - 2}")  # 扣除 acct 和 label
    62
"""

import time
import joblib
import os
import numpy as np
import pandas as pd
from Config.config import Config

class FeatureEngineer:
    """特徵工程器 - 從交易資料中提取 62 維特徵。
    
    此類別實作完整的特徵提取流程，包含：
    - High Activity 10x 採樣策略（針對訓練集）
    - 批次提取特徵並顯示進度
    - 快取機制以加速重複執行
    - 處理無交易帳戶的零填充
    
    Attributes:
        df_txn (pd.DataFrame): 交易資料，需包含 from_acct, to_acct, txn_amt 等欄位。
    
    Example:
        >>> engineer = FeatureEngineer(df_txn)
        >>> # 提取訓練集特徵（含採樣）
        >>> df_train = engineer.extract_train_features(df_alert)
        >>> # 提取測試集特徵
        >>> df_test = engineer.extract_test_features(df_predict)
    """
    
    def __init__(self, df_txn):
        """初始化特徵工程器。
        
        Args:
            df_txn (pd.DataFrame): 交易資料，需包含以下欄位：
                - from_acct: 轉出帳戶
                - to_acct: 轉入帳戶
                - txn_amt: 交易金額（已轉換為 TWD）
                - txn_date: 交易日期
                - txn_hour: 交易小時
                - is_self_txn: 是否為自轉交易
                - channel_type: 交易通路
                - currency_type: 幣別
                - from_acct_type: 轉出帳戶類型（1=玉山, 2=他行）
                - to_acct_type: 轉入帳戶類型（1=玉山, 2=他行）
        """
        self.df_txn = df_txn
    
    def extract_train_features(self, df_alert, use_cache=True):
        """提取訓練集特徵（含 High Activity 採樣策略）。
        
        此方法實作 High Activity 10x 採樣策略：
        1. 識別所有警示帳戶作為正樣本
        2. 計算所有非警示帳戶的交易量
        3. 從 Top 30% 高活躍帳戶中隨機採樣 10x 負樣本
        4. 提取所有選定帳戶的特徵
        5. 添加標籤欄位（label: 1=警示, 0=正常）
        
        Args:
            df_alert (pd.DataFrame): 警示帳戶清單，需包含 acct 欄位。
            use_cache (bool, optional): 是否使用快取。
                若為 True 且快取檔案存在，則直接載入快取。
                預設為 True。
        
        Returns:
            pd.DataFrame: 包含特徵和標籤的訓練資料，欄位包括：
                - acct: 帳戶 ID
                - 62 個特徵欄位（見模組 docstring）
                - label: 標籤（1=警示, 0=正常）
        
        Example:
            >>> engineer = FeatureEngineer(df_txn)
            >>> df_train = engineer.extract_train_features(df_alert, use_cache=True)
            >>> print(f"訓練樣本數: {len(df_train)}")
            >>> print(f"警示比例: {df_train['label'].mean():.2%}")
        """
        print("="*70)
        print(f"📊 [特徵工程] 準備訓練資料 - 策略: {Config.SAMPLING_STRATEGY}")
        
        # 檢查快取
        if use_cache and os.path.exists(Config.TRAIN_FEATURE_CACHE):
            print(f"✓ 載入訓練特徵快取: {Config.TRAIN_FEATURE_CACHE}")
            cache_data = joblib.load(Config.TRAIN_FEATURE_CACHE)
            df_features = cache_data['df_train']
            print(f"✓ 特徵數: {len(df_features.columns)-2}, 樣本數: {len(df_features):,}")
            print(f"✓ 警示比例: {df_features['label'].mean():.4%}")
            return df_features
        
        alert_accts = set(df_alert['acct'].unique())
        print(f"警示帳戶數: {len(alert_accts)}")
        
        # 計算所有玉山帳戶的交易量
        esun_out = self.df_txn[self.df_txn['from_acct_type'] == 1].groupby('from_acct').size()
        esun_in = self.df_txn[self.df_txn['to_acct_type'] == 1].groupby('to_acct').size()
        txn_counts = pd.concat([esun_out, esun_in]).groupby(level=0).sum().sort_values(ascending=False)
        
        # 排除警示帳戶
        non_alert_txn_counts = txn_counts[~txn_counts.index.isin(alert_accts)]
        
        # High Activity 採樣
        top_n = int(len(non_alert_txn_counts) * Config.TOP_PERCENTILE)
        high_activity_accts = non_alert_txn_counts.head(top_n).index.tolist()
        
        sample_size = min(len(high_activity_accts), len(alert_accts) * Config.NEGATIVE_MULTIPLIER)
        np.random.seed(Config.RANDOM_SEED)
        sampled_non_alert = np.random.choice(high_activity_accts, size=sample_size, replace=False)
        
        print(f"可選高活躍帳戶 (Top {Config.TOP_PERCENTILE*100:.0f}%): {len(high_activity_accts):,}")
        print(f"實際採樣負樣本: {len(sampled_non_alert):,}")
        
        train_accts = list(alert_accts) + list(sampled_non_alert)
        print(f"訓練帳戶總數: {len(train_accts):,} (警示:{len(alert_accts)} + 採樣:{len(sampled_non_alert)})")
        
        # 提取特徵
        df_features = self._extract_features_batch(train_accts, stage_name="訓練特徵提取")
        df_features['label'] = df_features['acct'].apply(lambda x: 1 if x in alert_accts else 0)
        
        print(f"✓ 特徵數: {len(df_features.columns)-2}, 警示比例: {df_features['label'].mean():.4%}")
        
        # 快取
        if use_cache:
            joblib.dump({'df_train': df_features}, Config.TRAIN_FEATURE_CACHE, compress=3)
            print(f"✓ 已快取至 {Config.TRAIN_FEATURE_CACHE}")
        
        return df_features
    
    def extract_test_features(self, df_predict, use_cache=True):
        """提取測試集特徵。
        
        為所有待預測帳戶提取相同的 62 維特徵向量。
        
        Args:
            df_predict (pd.DataFrame): 待預測帳戶清單，需包含 acct 欄位。
            use_cache (bool, optional): 是否使用快取。預設為 True。
        
        Returns:
            pd.DataFrame: 包含特徵的測試資料，欄位包括：
                - acct: 帳戶 ID
                - 62 個特徵欄位
        
        Example:
            >>> engineer = FeatureEngineer(df_txn)
            >>> df_test = engineer.extract_test_features(df_predict)
            >>> print(f"測試樣本數: {len(df_test)}")
        """
        print("="*70)
        print("🎯 [特徵工程] 準備測試資料")
        
        # 檢查快取
        if use_cache and os.path.exists(Config.TEST_FEATURE_CACHE):
            print(f"✓ 載入測試特徵快取: {Config.TEST_FEATURE_CACHE}")
            df_features = joblib.load(Config.TEST_FEATURE_CACHE)
            print(f"✓ 特徵數: {len(df_features.columns)-1}, 樣本數: {len(df_features):,}")
            return df_features
        
        test_accts = df_predict['acct'].tolist()
        print(f"測試帳戶數: {len(test_accts)}")
        
        # 提取特徵
        df_features = self._extract_features_batch(test_accts, stage_name="測試特徵提取")
        
        # 快取
        if use_cache:
            joblib.dump(df_features, Config.TEST_FEATURE_CACHE, compress=3)
            print(f"✓ 已快取至 {Config.TEST_FEATURE_CACHE}")
        
        return df_features
    
    def _extract_features_batch(self, account_list, stage_name="特徵提取"):
        """批次提取帳戶特徵並顯示進度。
        
        此方法批次處理帳戶清單，為每個帳戶提取特徵，
        並在處理過程中顯示進度資訊（每 1000 個帳戶更新一次）。
        
        Args:
            account_list (list): 帳戶 ID 清單。
            stage_name (str, optional): 階段名稱，用於日誌輸出。
                預設為 "特徵提取"。
        
        Returns:
            pd.DataFrame: 包含所有帳戶特徵的 DataFrame。
        
        Note:
            - 每 1000 個帳戶顯示一次進度
            - 顯示處理速度（accts/s）和預估剩餘時間
            - 對於無交易帳戶，使用 _get_empty_features() 填充零值
        """
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
        """提取單個帳戶的 62 維特徵向量。
        
        此方法為單一帳戶計算所有特徵，包含：
        
        1. 基本交易統計 (5 個):
           - out_count, in_count, total_count: 轉出/轉入/總交易次數
           - txn_ratio: 轉出轉入比例
           - in_out_diff: 轉出轉入次數差異
        
        2. 金額特徵 (23 個):
           - out_amt_*: 轉出金額統計（sum, mean, std, max, min, median, cv, q25, q75, iqr）
           - in_amt_*: 轉入金額統計（sum, mean, median）
           - amt_in_out_ratio: 轉入轉出金額比例
           - amt_balance: 轉入轉出金額差
        
        3. 時間特徵 (13 個):
           - date_range: 交易日期範圍
           - txn_velocity: 交易速度（次數/天）
           - active_days, active_day_ratio: 活躍天數與比例
           - night_txn_count, night_txn_ratio: 夜間交易統計
           - work_hour_count, work_hour_ratio: 工作時段交易統計
           - first_week_ratio, last_week_ratio: 首週末週交易比例
           - first_last_week_diff: 首末週差異
           - burst_score, hour_entropy: 爆發分數與時間熵
        
        4. 對手方特徵 (4 個):
           - unique_from, unique_to: 唯一轉入/轉出對象數
           - unique_counterparties: 唯一對手方總數
           - counterparty_diversity: 對手方多樣性
        
        5. 交易類型特徵 (11 個):
           - self_txn_count, self_txn_ratio: 自轉交易統計
           - unique_channels, channel_diversity: 通路多樣性
           - foreign_currency_*: 外幣交易統計
           - cross_bank_*: 跨行交易統計
        
        6. 風險指標 (14 個):
           - large_txn_*: 大額交易統計（P90 以上）
           - small_txn_*: 小額交易統計（P10 以下）
           - out_concentration: 轉出集中度
           - in_concentration: 轉入集中度
           - hour_entropy: 交易時段熵值
        
        Args:
            out_txns (pd.DataFrame): 該帳戶的轉出交易資料。
            in_txns (pd.DataFrame): 該帳戶的轉入交易資料。
            all_txns (pd.DataFrame): 該帳戶的所有交易資料。
        
        Returns:
            dict: 包含 62 個特徵的字典，key 為特徵名稱，value 為特徵值。
        
        Note:
            - 對於無轉出或轉入交易的情況，相關特徵填充為 0
            - 分母可能為 0 的計算會加上 1e-6 避免除零錯誤
        """
        features = {}
        
        # ===== 1. 基本交易統計 =====
        features['out_count'] = len(out_txns)
        features['in_count'] = len(in_txns)
        features['total_count'] = len(all_txns)
        features['txn_ratio'] = features['out_count'] / max(features['in_count'], 1)
        features['in_out_diff'] = abs(features['out_count'] - features['in_count'])
        
        # ===== 2. 金額特徵 =====
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
        
        features['night_txn_count'] = len(all_txns[(all_txns['txn_hour'] < 6) | (all_txns['txn_hour'] >= 22)])
        features['night_txn_ratio'] = features['night_txn_count'] / features['total_count']
        features['work_hour_count'] = len(all_txns[(all_txns['txn_hour'] >= 9) & (all_txns['txn_hour'] < 18)])
        features['work_hour_ratio'] = features['work_hour_count'] / features['total_count']
        
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
        
        if len(out_txns) > 0:
            threshold_10 = out_txns['txn_amt'].quantile(0.1)
            features['small_txn_count'] = len(out_txns[out_txns['txn_amt'] < threshold_10])
            features['small_txn_ratio'] = features['small_txn_count'] / len(out_txns)
        else:
            features['small_txn_count'] = 0
            features['small_txn_ratio'] = 0
        
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
        
        daily_counts = all_txns.groupby('txn_date').size()
        if len(daily_counts) > 0:
            features['max_daily_txns'] = daily_counts.max()
            features['avg_daily_txns'] = daily_counts.mean()
            features['std_daily_txns'] = daily_counts.std()
            features['burst_score'] = features['max_daily_txns'] / (features['avg_daily_txns'] + 1e-6)
        else:
            for k in ['max_daily_txns','avg_daily_txns','std_daily_txns','burst_score']:
                features[k] = 0
        
        hour_dist = all_txns['txn_hour'].value_counts(normalize=True)
        features['hour_entropy'] = -np.sum(hour_dist * np.log(hour_dist + 1e-10))
        
        features['unique_dates'] = all_txns['txn_date'].nunique()
        features['date_coverage'] = features['unique_dates'] / features['date_range']
        
        return features
    
    def _get_empty_features(self):
        """返回空特徵字典（用於無交易帳戶）。
        
        當帳戶沒有任何交易記錄時，回傳所有特徵值為 0 的字典。
        確保特徵維度一致性，避免後續模型訓練出錯。
        
        Returns:
            dict: 包含 62 個特徵的字典，所有值均為 0。
        
        Note:
            此方法列出所有特徵名稱，用於：
            1. 處理無交易帳戶的零填充
            2. 作為完整特徵清單的參考文件
        """
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
