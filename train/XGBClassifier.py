import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader

# ---------- 1. 数据集封装 ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, :5].values.astype(np.float32)
        # 三分类标签 {-1,0,1} → {0,1,2}
        y_raw = df.iloc[:, 5:].values.astype(np.int64)
        self.y = y_raw + 1          # 关键：-1→0, 0→1, 1→2
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


train_loader = DataLoader(TimeSeriesDataset('solve_generated_data.csv'),
                          batch_size=4096, shuffle=False)
valid_loader = DataLoader(TimeSeriesDataset('solve_generated_data_e4.csv'),
                          batch_size=4096, shuffle=False)
test_loader = DataLoader(TimeSeriesDataset('solve_generated_data_e5.csv'),
                         batch_size=4096, shuffle=False)

X_train, y_train = load_all_data(train_loader)
X_valid, y_valid = load_all_data(valid_loader)
X_test, y_test = load_all_data(test_loader)

# ---------- 2. 三分类模型 ----------
class MultiXGBClassifier:
    def __init__(self,
                 max_depth=20,
                 n_estimators=1000,
                 learning_rate=0.01):
        self.models = []
        self.params = dict(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective='multi:softprob',
            num_class=3,
            tree_method='hist',
            device='cuda',
            random_state=42
        )

    def train(self, X, y):
        self.models = [
            xgb.XGBClassifier(**self.params).fit(X, y[:, col])
            for col in range(y.shape[1])
        ]

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds - 1   # 映射回 {-1,0,1}


model = MultiXGBClassifier()
model.train(X_train, y_train)

# ---------- 3. 评估 ----------
def evaluate(model, X, y, phase='Valid'):
    preds = model.predict(X)          # shape (N, 5)
    # 逐行比较，5 个全对为 True，否则 False
    correct_rows = (preds == y).all(axis=1)
    acc = correct_rows.mean()
    print(f"{phase} Accuracy (all-5-correct): {acc*100:.2f}%")
    return acc

evaluate(model, X_valid, y_valid, phase='Valid')
evaluate(model, X_test, y_test, phase='Test')

# ---------- 4. 保存 ----------
os.makedirs('log', exist_ok=True)
for idx, m in enumerate(model.models):
    m.save_model(f'log/xgb_cls_target{idx}.json')