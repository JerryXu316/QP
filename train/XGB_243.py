import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader

# ---------- 1. 数据集封装 ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, :5].values.astype(np.float32)
        y_raw = df.iloc[:, 5:].values.astype(np.int64)
        self.y = y_raw + 1          # {-1,0,1} -> {0,1,2}
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_all_data(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.vstack(xs), np.vstack(ys)


# ---------- 2. 读取数据 ----------
train_loader = DataLoader(TimeSeriesDataset('solve_generated_data.csv'),
                          batch_size=4096, shuffle=False)
valid_loader = DataLoader(TimeSeriesDataset('solve_generated_data_e4.csv'),
                          batch_size=4096, shuffle=False)
test_loader = DataLoader(TimeSeriesDataset('solve_generated_data_e5.csv'),
                         batch_size=4096, shuffle=False)

X_train, y_train_multi = load_all_data(train_loader)
X_valid, y_valid_multi = load_all_data(valid_loader)
X_test,  y_test_multi  = load_all_data(test_loader)

# ---------- 3. 把 5 维标签编码成 243 类 ----------
def encode_243(y_multi):
    return np.ravel_multi_index(y_multi.T, (3, 3, 3, 3, 3))   # 0..242

y_train_243 = encode_243(y_train_multi)
y_valid_243 = encode_243(y_valid_multi)
y_test_243  = encode_243(y_test_multi)

# ---------- 4. 补齐 243 类 ----------
K = 243
# 生成 243 个“影子”样本，标签 0..242 各一次
X_dummy = np.zeros((K, X_train.shape[1]), dtype=np.float32)
y_dummy = np.arange(K, dtype=np.int64)

# 合并
X_full = np.vstack([X_train, X_dummy])
y_full = np.concatenate([y_train_243, y_dummy])

# 权重：真实样本 1，影子样本 0
w_full = np.ones(len(y_full))
w_full[len(y_train_243):] = 0.0

# ---------- 5. 243 类 XGBoost ----------
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=K,
    max_depth=12,
    n_estimators=800,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    device='cuda',
    eval_metric='mlogloss',
    random_state=42
)

model.fit(
    X_full, y_full,
    sample_weight=w_full,               # 影子样本权重为 0
    eval_set=[(X_valid, y_valid_243)],
    verbose=100
)

# ---------- 6. 行级指标 ----------
def calculate_metrics_243(model, X, y_multi):
    proba = model.predict_proba(X)              # (N,243)
    y_true = encode_243(y_multi)

    # 排序
    order = np.argsort(proba, axis=1)[:, ::-1]
    ranks = np.array([np.where(order[i] == y_true[i])[0][0] + 1
                      for i in range(len(y_true))])

    top1 = np.mean(ranks == 1)
    top3 = np.mean(ranks <= 3)
    exp_rank = ranks.mean()

    print(f"Row-wise Top-1: {top1*100:.2f}%")
    print(f"Row-wise Top-3: {top3*100:.2f}%")
    print(f"Row-wise Expected Rank: {exp_rank:.2f}")


# ---------- 7. 评估 ----------
print("=== Validation ===")
calculate_metrics_243(model, X_valid, y_valid_multi)

print("=== Test ===")
calculate_metrics_243(model, X_test, y_test_multi)

# ---------- 8. 保存 ----------
os.makedirs('log', exist_ok=True)
model.save_model('log/xgb_243cls_full.json')