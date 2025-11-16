"""配置模組 - 超參數與路徑設定。

此模組提供集中化的配置管理，包含所有實驗相關的參數設定。

Classes:
    Config: 配置類別，包含所有超參數、路徑、採樣策略等設定。

Example:
    >>> from Config.config import Config
    >>> print(Config.RANDOM_SEED)
    42
    >>> print(Config.EXCHANGE_RATES['USD'])
    30
"""

class Config:
    """配置類別 - 集中管理所有超參數與路徑設定。
    
    此類別包含訓練、預測、特徵工程所需的所有配置參數。
    所有屬性均為類別變數，可直接透過 Config.ATTRIBUTE 存取。
    
    Attributes:
        DATA_PATH (str): 競賽資料目錄路徑。
        OUTPUT_PATH (str): 輸出結果目錄路徑。
        TRAIN_FEATURE_CACHE (str): 訓練特徵快取檔名。
        TEST_FEATURE_CACHE (str): 測試特徵快取檔名。
        MODEL_CACHE (str): 模型快取檔名。
        RANDOM_SEED (int): 隨機種子，確保實驗可重現。
        SAMPLING_STRATEGY (str): 採樣策略名稱。
        TOP_PERCENTILE (float): 高活躍帳戶百分比（0.3 = Top 30%）。
        NEGATIVE_MULTIPLIER (int): 負樣本倍數（10 表示負樣本為正樣本的 10 倍）。
        EXCHANGE_RATES (dict): 幣別匯率字典，key 為幣別代碼，value 為對 TWD 匯率。
        LGBM_PARAMS (dict): LightGBM 超參數。
        XGB_PARAMS (dict): XGBoost 超參數。
        CAT_PARAMS (dict): CatBoost 超參數。
        N_SPLITS (int): 交叉驗證的 fold 數。
        THRESHOLD_MIN (float): 閾值搜索最小值。
        THRESHOLD_MAX (float): 閾值搜索最大值。
        THRESHOLD_STEP (float): 閾值搜索步長。
        N_JOBS (int): 平行處理的 CPU 核心數（-1 表示使用所有核心）。
    
    Example:
        >>> from Config.config import Config
        >>> # 取得採樣策略
        >>> print(Config.SAMPLING_STRATEGY)
        'high_activity_10x'
        >>> # 取得 LightGBM 參數
        >>> lgb_params = Config.LGBM_PARAMS
        >>> print(lgb_params['learning_rate'])
        0.05
    """
    # ==================== 路徑設定 ====================
    DATA_PATH = './40_初賽資料_V3 1/初賽資料/'
    OUTPUT_PATH = './output/'
    
    TRAIN_FEATURE_CACHE = "features_train.pkl"
    TEST_FEATURE_CACHE = "features_test.pkl"
    MODEL_CACHE = "ensemble_models.pkl"
    
    # ==================== 隨機種子 ====================
    RANDOM_SEED = 42
    
    # ==================== 採樣策略 ====================
    SAMPLING_STRATEGY = 'high_activity_10x'
    TOP_PERCENTILE = 0.3      # 從 Top 30% 高活躍帳戶中採樣
    NEGATIVE_MULTIPLIER = 10  # 負樣本為正樣本的 10 倍
    
    # ==================== 特徵工程 ====================
    # 幣別匯率（相對於 TWD）
    EXCHANGE_RATES = {
        'AUD': 19.6, 'CAD': 21.7, 'CHF': 37.74, 'CNY': 4.2,
        'EUR': 35, 'GBP': 40.2, 'HKD': 3.82, 'JPY': 0.2,
        'NZD': 17.5, 'SEK': 3.15, 'SGD': 23.24, 'THB': 0.92,
        'TWD': 1, 'USD': 30, 'ZAR': 1.72
    }
    
    # ==================== 模型超參數 ====================
    
    # LightGBM
    LGBM_PARAMS = {
        'objective': 'binary',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'random_state': RANDOM_SEED,
        'verbosity': -1
    }
    
    # XGBoost
    XGB_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'random_state': RANDOM_SEED,
        'verbosity': 0
    }
    
    # CatBoost
    CAT_PARAMS = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'random_seed': RANDOM_SEED,
        'verbose': False
    }
    
    # ==================== 交叉驗證 ====================
    N_SPLITS = 5  # 5-Fold Stratified CV
    
    # ==================== 閾值搜索 ====================
    THRESHOLD_MIN = 0.1
    THRESHOLD_MAX = 0.9
    THRESHOLD_STEP = 0.01
    
    # ==================== 資源配置 ====================
    N_JOBS = -1  # 使用所有 CPU 核心
