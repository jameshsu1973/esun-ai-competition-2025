"""
主程式 - 模組化執行範例

此程式展示如何使用各個模組進行完整的訓練和預測流程。
如需快速執行，請直接使用 ImprovedFraudDetector.py
"""

import sys
import os

# 確保可以 import 專案模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Config.config import Config
from Preprocess.data_loader import DataLoader
from Preprocess.feature_engineering import FeatureEngineer
from Model.ensemble_trainer import EnsembleTrainer
from Model.predictor import Predictor

def main():
    """主程式流程"""
    
    print("\n" + "="*70)
    print("  玉山 AI 挑戰賽 - 詐欺交易偵測系統")
    print("  模組化執行版本")
    print("="*70)
    
    # ==================== 階段1: 資料載入 ====================
    loader = DataLoader(Config.DATA_PATH)
    df_txn, df_alert, df_predict = loader.load_all()
    
    # ==================== 階段2: 特徵工程 ====================
    engineer = FeatureEngineer(df_txn)
    
    # 提取訓練集特徵
    df_train = engineer.extract_train_features(df_alert, use_cache=True)
    
    # 提取測試集特徵
    df_test = engineer.extract_test_features(df_predict, use_cache=True)
    
    # ==================== 階段3: 模型訓練 ====================
    trainer = EnsembleTrainer(Config)
    trainer.train(df_train)
    
    # ==================== 階段4: 模型預測 ====================
    predictor = Predictor(
        models=trainer.models,
        scaler=trainer.scaler,
        threshold=trainer.threshold,
        feature_cols=trainer.feature_cols
    )
    
    predictions, probabilities = predictor.predict(df_test)
    
    # ==================== 階段5: 儲存結果 ====================
    predictor.save(
        predictions=predictions,
        probabilities=probabilities,
        account_ids=df_predict['acct']
    )
    
    print("\n" + "="*70)
    print("✅ 完成！")
    print("="*70)

if __name__ == "__main__":
    main()
