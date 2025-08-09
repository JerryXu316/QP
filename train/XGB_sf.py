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
            objective='multi:softprob',  # 使用 softmax 预测概率
            num_class=3,                 # 三分类问题
            tree_method='hist',
            device='cuda',  # 如果有 CUDA 可以用
            random_state=42
        )

    def train(self, X, y):
        self.models = [
            xgb.XGBClassifier(**self.params).fit(X, y[:, col])  # 每个目标训练一个模型
            for col in range(y.shape[1])
        ]

    def predict_proba(self, X):
        # 输出每个类别的预测概率
        preds = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        return preds  # shape: (N, 3)

    def predict(self, X):
        # 输出每个样本的最大概率解组
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)  # 返回最大概率的解组

# ---------- 3. 计算指标 ----------
import itertools
import numpy as np

# ---------- 3. 计算“整行”指标 ----------
def _all_possible_5tuple_probs(probs):
    """
    probs : list of 5 (N,3) ndarray
    返回 (N, 3^5) ndarray，每一行是某一条 5-tuple 的联合概率
    """
    N = probs[0].shape[0]
    # 5 个位置各自的 3 个概率
    p0, p1, p2, p3, p4 = probs           # 每个 (N,3)
    # 预先生成 3^5=243 种组合
    idx_list = list(itertools.product([0,1,2], repeat=5))   # len=243
    joint = np.zeros((N, len(idx_list)))
    for k, (i0,i1,i2,i3,i4) in enumerate(idx_list):
        joint[:, k] = (p0[:,i0] * p1[:,i1] * p2[:,i2] * p3[:,i3] * p4[:,i4])
    return joint, idx_list       # joint shape (N,243), idx_list list of tuple


def calculate_metrics(model, X, y, top_n=3):
    """
    y : (N,5) 真实标签 {0,1,2}
    计算：
        Top-1 Accuracy : 联合概率最大的 5-tuple 恰好是真实值的比例
        Top-N Accuracy : 真实值落在联合概率前 N 个 5-tuple 中的比例
        Expected Rank  : 真实 5-tuple 在所有 243 种可能中的平均排序
    """
    # 1) 拿到 5 个独立模型各自的 (N,3) softmax 概率
    probs = [m.predict_proba(X) for m in model.models]   # len=5, each (N,3)

    # 2) 计算 3^5=243 种组合的联合概率
    joint, idx_list = _all_possible_5tuple_probs(probs)   # joint shape (N,243)

    # 3) 把真实标签转成 tuple，便于比较
    y_tuples = [tuple(row) for row in y]                  # list of len N

    top1_hit, topn_hit, rank_sum = 0, 0, 0
    for i in range(len(y)):
        true_tuple = y_tuples[i]
        # 排序索引（概率从高到低）
        order = np.argsort(joint[i])[::-1]
        # 真实 tuple 在 243 种里的索引
        true_idx = idx_list.index(true_tuple)
        rank = np.where(order == true_idx)[0][0] + 1   # 1-based
        if rank == 1:
            top1_hit += 1
        if rank <= top_n:
            topn_hit += 1
        rank_sum += rank

    N = len(y)
    top1_acc = top1_hit / N
    topn_acc = topn_hit / N
    exp_rank = rank_sum / N

    print(f"Row-wise Top-1: {top1_acc*100:.2f}%")
    print(f"Row-wise Top-{top_n}: {topn_acc*100:.2f}%")
    print(f"Row-wise Expected Rank: {exp_rank:.2f}")


# ---------- 4. 训练与评估 ----------
model = MultiXGBClassifier()
model.train(X_train, y_train)

# 评估
print("=== Validation ===")
calculate_metrics(model, X_valid, y_valid, top_n=3)
print("=== Test ===")
calculate_metrics(model, X_test, y_test, top_n=3)

# ---------- 5. 保存模型 ----------
os.makedirs('log', exist_ok=True)
for idx, m in enumerate(model.models):
    m.save_model(f'log/xgb_cls_target{idx}.json')
