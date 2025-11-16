"""資料載入與預處理模組。

此模組負責載入競賽原始資料並進行基本預處理，包含：
- 載入交易資料、警示帳戶、待預測帳戶
- 解析交易時間為小時特徵
- 統一幣別轉換為 TWD（台幣）

Classes:
    DataLoader: 資料載入器，提供統一的資料載入介面。

Example:
    >>> from Config.config import Config
    >>> from Preprocess.data_loader import DataLoader
    >>> loader = DataLoader(Config.DATA_PATH)
    >>> df_txn, df_alert, df_predict = loader.load_all()
    >>> print(f"交易筆數: {len(df_txn):,}")
"""

import pandas as pd
from Config.config import Config

class DataLoader:
    """資料載入器 - 載入並預處理競賽資料。
    
    此類別負責從指定路徑載入競賽的三個主要資料檔案，
    並進行基本的預處理作業（時間解析、幣別轉換）。
    
    Attributes:
        data_path (str): 資料目錄路徑。
    
    Example:
        >>> loader = DataLoader('./40_初賽資料_V3 1/初賽資料/')
        >>> df_txn, df_alert, df_predict = loader.load_all()
        >>> print(df_txn.columns.tolist())
        ['from_acct', 'to_acct', 'txn_amt', 'currency_type', ...]
    """
    
    def __init__(self, data_path=None):
        """初始化資料載入器。
        
        Args:
            data_path (str, optional): 資料目錄路徑。
                若未指定則使用 Config.DATA_PATH。
        """
        self.data_path = data_path or Config.DATA_PATH
    
    def load_all(self):
        """載入所有資料並進行基本預處理。
        
        載入三個主要資料檔案：
        1. acct_transaction.csv - 交易明細資料
        2. acct_alert.csv - 警示帳戶標籤
        3. acct_predict.csv - 待預測帳戶清單
        
        預處理步驟：
        1. 解析 txn_time 為小時特徵（txn_hour）
        2. 使用 Config.EXCHANGE_RATES 將所有幣別轉換為 TWD
        
        Returns:
            tuple: 包含三個 DataFrame 的元組
                - df_txn (pd.DataFrame): 交易資料，已完成幣別轉換
                - df_alert (pd.DataFrame): 警示帳戶清單
                - df_predict (pd.DataFrame): 待預測帳戶清單
        
        Example:
            >>> loader = DataLoader()
            >>> df_txn, df_alert, df_predict = loader.load_all()
            >>> print(f"警示帳戶數: {len(df_alert)}")
            >>> print(f"待測帳戶數: {len(df_predict)}")
        """
        print("="*70)
        print("📂 [資料載入] 載入競賽資料集")
        
        # 載入資料
        df_txn = pd.read_csv(f'{self.data_path}acct_transaction.csv')
        df_alert = pd.read_csv(f'{self.data_path}acct_alert.csv')
        df_predict = pd.read_csv(f'{self.data_path}acct_predict.csv')
        
        # 預處理時間
        df_txn['txn_hour'] = pd.to_datetime(
            df_txn['txn_time'], format='%H:%M:%S', errors='coerce'
        ).dt.hour.fillna(12)
        
        # 幣別轉換為 TWD
        df_txn['txn_amt'] = df_txn.apply(
            lambda row: row['txn_amt'] * Config.EXCHANGE_RATES.get(row['currency_type'], 1), 
            axis=1
        )
        
        print(f"✓ 交易資料: {len(df_txn):,} 筆")
        print(f"✓ 警示帳戶: {len(df_alert):,} 個")
        print(f"✓ 待測帳戶: {len(df_predict):,} 個")
        print(f"✓ 已完成幣別轉換（統一為 TWD）")
        
        return df_txn, df_alert, df_predict
