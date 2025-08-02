import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader
import joblib

# -------------------------------------------------
# 1. 数据封装（标签映射 0/1/2）
# -------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, :5].values.astype(np.float32)
        y_raw = df.iloc[:, 5:].values.astype(np.int64)
        self.y = y_raw + 1  # -1→0, 0→1, 1→2
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_all(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.vstack(xs), np.vstack(ys)


train_loader = DataLoader(TimeSeriesDataset('solve_generated_data.csv'),
                          batch_size=4096, shuffle=True)
valid_loader = DataLoader(TimeSeriesDataset('solve_generated_data_e4.csv'),
                          batch_size=4096, shuffle=False)
test_loader = DataLoader(TimeSeriesDataset('solve_generated_data_e5.csv'),
                         batch_size=4096, shuffle=False)

X_train, y_train = load_all(train_loader)
X_valid, y_valid = load_all(valid_loader)
X_test, y_test = load_all(test_loader)

# -------------------------------------------------
# 2. 5 个 XGBClassifier（三分类）
# -------------------------------------------------
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
            device='cuda',  # 仅此一行即可
            random_state=42
        )

    def train(self, X, y):
        self.models = [
            xgb.XGBClassifier(**self.params).fit(X, y[:, col])
            for col in range(y.shape[1])
        ]

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds - 1  # 映射回 {-1,0,1}


model = MultiXGBClassifier()
model.train(X_train, y_train)

# -------------------------------------------------
# 3. 评估（整行全对）
# -------------------------------------------------
def evaluate(model, X, y, phase='Valid'):
    preds = model.predict(X)
    correct_rows = (preds == y - 1).all(axis=1)  # y-1 把 0/1/2 变回 -1/0/1
    acc = correct_rows.mean()
    print(f"{phase} 整行全对准确率: {acc:.4f}")
    return acc


evaluate(model, X_valid, y_valid, phase='Valid')
evaluate(model, X_test, y_test, phase='Test')

# -------------------------------------------------
# 4. 保存
# -------------------------------------------------
os.makedirs('model', exist_ok=True)
joblib.dump(train_loader.dataset.scaler, 'model/scaler.pkl')
for idx, m in enumerate(model.models):
    m.save_model(f'model/xgb_cls_target{idx}.json')