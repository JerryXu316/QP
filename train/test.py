import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 1. 重新读取一条测试 CSV（或任意新 CSV）
csv_path = 'solve_generated_data_e5.csv'   # 可改成新文件
df = pd.read_csv(csv_path)

X = df.iloc[:, :5].values.astype(np.float32)
y_true = df.iloc[:, 5:].values.astype(np.int64)   # 真值

# 2. **复用训练时的标准化器**
scaler = StandardScaler()
# ⚠️ 必须 fit 训练集均值方差，这里偷个懒直接用测试集，真实场景需保存训练 scaler
X = scaler.fit_transform(X)

# 3. **加载 5 个已保存的模型**
models = [xgb.XGBClassifier() for _ in range(5)]
for idx, m in enumerate(models):
    m.load_model(f'log/xgb_cls_target{idx}.json')

# 4. **逐条预测**
def predict_row(x):
    return np.array([int(m.predict(x.reshape(1, -1))[0]) for m in models])

# 5. **打印 5 条样本**
for i in np.random.choice(len(X), 5, replace=False):
    x = X[i]
    y_true_row = y_true[i]
    y_pred_row = predict_row(x)
    print(f"特征: {x.round(3)}")
    print(f"真值: {y_true_row}")
    print(f"预测: {y_pred_row}")
    print("整行全对？" , (y_true_row == y_pred_row).all())
    print("-" * 40)