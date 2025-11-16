# 快速開始指南

## 安裝步驟

### 1. 克隆專案
```bash
git clone [您的 GitHub 連結]
cd AI_cup
```

### 2. 安裝依賴
```bash
pip install -r requirements.txt
```

### 3. 準備資料
將競賽資料放置於 `./40_初賽資料_V3 1/初賽資料/` 目錄：
```
40_初賽資料_V3 1/
└── 初賽資料/
    ├── acct_transaction.csv
    ├── acct_alert.csv
    └── acct_predict.csv
```

## 執行方式

### 方法1: 快速執行（推薦）
直接執行整合主程式：
```bash
python ImprovedFraudDetector.py
```

### 方法2: 模組化執行
```bash
python main.py
```

## 輸出結果

執行完成後，結果將儲存在 `output/` 目錄：
- `submission_improved.csv` - 競賽提交格式（不含機率）
- `submission_improved_with_prob.csv` - 含預測機率版本
- `threshold_analysis.png` - 閾值分析圖表

## 重新訓練

如需重新提取特徵（不使用快取）：

修改 `ImprovedFraudDetector.py` 或 `main.py` 中的參數：
```python
df_train = detector.prepare_training_data(use_cache=False)
df_test = detector.prepare_test_data(use_cache=False)
```

## 調整超參數

編輯 `Config/config.py` 修改模型參數：
- `LGBM_PARAMS` - LightGBM 參數
- `XGB_PARAMS` - XGBoost 參數
- `CAT_PARAMS` - CatBoost 參數
- `N_SPLITS` - 交叉驗證 Fold 數
- `THRESHOLD_MIN/MAX/STEP` - 閾值搜索範圍

## 系統需求

- Python 3.11.7
- RAM: 至少 8GB

## 問題排除

### 記憶體不足
減少訓練樣本數量，修改 `Config/config.py`：
```python
NEGATIVE_MULTIPLIER = 5  # 從 10 改為 5
```

### 找不到資料
確認資料路徑設定正確：
```python
DATA_PATH = './40_初賽資料_V3 1/初賽資料/'
```

