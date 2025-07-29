import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# 1. 数据集封装
# ------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, :5].values.astype(np.float32)
        # 标签仍是连续值，后续再离散化
        self.y = df.iloc[:, 5:].values.astype(np.float32)
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

# ------------------------------------------------------------
# 2. 回归模型（连续输出）
# ------------------------------------------------------------
class XGBoostModel:
    def __init__(self,
                 max_depth=20,
                 n_estimators=1000,
                 learning_rate=0.01):
        base = xgb.XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            tree_method='hist',
            device='cuda',
            random_state=42
        )
        self.model = MultiOutputRegressor(base)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


model = XGBoostModel()
model.train(X_train, y_train)

# ------------------------------------------------------------
# 3. 评估：四舍五入后计算准确率
# ------------------------------------------------------------
def evaluate(model, X, y, phase='Valid'):
    preds_cont = model.predict(X)                   # 连续值
    preds_disc = np.round(preds_cont).astype(int)   # 四舍五入
    preds_disc = np.clip(preds_disc, -1, 1)         # 保险到 {-1,0,1}
    y_disc = np.round(y).astype(int)                # 真值也转成离散

    # 单列准确率
    col_acc = (preds_disc == y_disc).mean(axis=0)
    # 整行全对准确率
    row_acc = (preds_disc == y_disc).all(axis=1).mean()

    print(f"{phase} 每列准确率: {col_acc.round(3)}")
    print(f"{phase} 平均列准确率: {col_acc.mean():.3f}")
    print(f"{phase} 整行全对准确率: {row_acc:.3f}")
    return row_acc


evaluate(model, X_valid, y_valid, phase='Valid')
evaluate(model, X_test, y_test, phase='Test')

# ------------------------------------------------------------
# 4. 保存模型 & 标准化器
# ------------------------------------------------------------
os.makedirs('log', exist_ok=True)

# 保存 scaler
import joblib
joblib.dump(train_loader.dataset.scaler, 'model/scaler.pkl')

# 保存 5 个子模型
for idx, est in enumerate(model.model.estimators_):
    est.save_model(f'model/xgb_target{idx}.json')