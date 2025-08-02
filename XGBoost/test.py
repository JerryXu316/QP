import os, gc, torch
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader
import joblib, optuna
from sklearn.model_selection import KFold

# 1. 数据封装（标签映射 0/1/2）
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, :5].values.astype(np.float32)
        y_raw = df.iloc[:, 5:].values.astype(np.int64)
        self.y = y_raw + 1
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

# 2. 整行全对目标 + 内存清零
def objective(trial):
    params = {
        'max_depth'        : trial.suggest_int('max_depth', 10, 30),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'objective'        : 'multi:softprob',
        'num_class'        : 3,
        'tree_method'      : 'hist',
        'device'           : 'cuda',
        'n_estimators'     : 600,
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X_train):
        fold_models = []
        for col in range(y_train.shape[1]):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train[train_idx], y_train[train_idx, col])
            fold_models.append(clf)
            del clf

        preds_fold = np.column_stack([
            m.predict(X_train[val_idx]) for m in fold_models
        ]) - 1
        row_acc = (preds_fold == y_train[val_idx] - 1).all(axis=1).mean()
        cv_scores.append(row_acc)
        torch.cuda.empty_cache()

    return np.mean(cv_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, show_progress_bar=True)
print("最佳参数:", study.best_params)

# 3. 用最佳参数重新训练 + 最终清理
best = study.best_params
best.update({'objective':'multi:softprob','num_class':3,
             'tree_method':'hist','device':'cuda','n_estimators':1000})
best_models = [xgb.XGBClassifier(**best).fit(X_train, y_train[:, c])
               for c in range(y_train.shape[1])]

preds = np.column_stack([m.predict(X_test) for m in best_models]) - 1
final_acc = (preds == y_test - 1).all(axis=1).mean()
print("测试集整行全对准确率:", final_acc)

# 4. 仅保存最终文件 + 彻底清理
joblib.dump(train_loader.dataset.scaler, 'model/scaler_opt.pkl')
for idx, m in enumerate(best_models):
    m.save_model(f'model/xgb_cls_target{idx}_opt.json')

del train_loader, valid_loader, test_loader, best_models
gc.collect()
torch.cuda.empty_cache()