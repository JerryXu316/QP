# evaluate_all_top.py
import torch
import pandas as pd
from dnn_softmax import TimeSeriesDataset, NonlinearDecisionNet
from torch.utils.data import DataLoader

MODEL_PATH = 'model_dnn/best_5x3.pth'
TEST_CSV   = 'solve_generated_data_e5.csv'
BATCH_SIZE = 2048
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据
test_ds   = TimeSeriesDataset(TEST_CSV)
loader    = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# 2. 模型
model = NonlinearDecisionNet(depth=15, base_ch=512, dropout=0.2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. 一次性计算 Top-1~Top-243 及 期望排名
@torch.no_grad()
def calc_all_top(model, loader):
    all_logits, all_y = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        all_logits.append(model(x))  # (N,5,3)
        all_y.append(y)              # (N,5)

    logits = torch.cat(all_logits, 0)
    y_true = torch.cat(all_y, 0)

    # 联合概率 (N,243)
    probs = torch.softmax(logits, dim=-1)
    joint = torch.ones(probs.shape[0], 243, device=DEVICE)
    idx = 0
    for t0, t1, t2, t3, t4 in torch.cartesian_prod(*[torch.arange(3)]*5):
        joint[:, idx] = probs[:, 0, t0] * probs[:, 1, t1] * probs[:, 2, t2] * probs[:, 3, t3] * probs[:, 4, t4]
        idx += 1

    y_idx = (
        y_true[:, 0] * 3**4 + y_true[:, 1] * 3**3 +
        y_true[:, 2] * 3**2 + y_true[:, 3] * 3**1 + y_true[:, 4]
    )

    order = torch.argsort(joint, dim=1, descending=True)
    ranks = (order == y_idx.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1  # 1-based

    # Top-1~Top-243 准确率
    top_k_acc = [(ranks <= k).float().mean().item() for k in range(1, 244)]
    # 期望排名
    expected_rank = ranks.float().mean().item()

    return top_k_acc, expected_rank

# 4. 执行
top_k, exp_rank = calc_all_top(model, loader)

# 5. 打印 & 保存
print(f'Expected Rank: {exp_rank:.4f}')
print('Top-1 ~ Top-243 准确率（CSV 一行）')
print(','.join(f'{acc:.4f}' for acc in top_k))

# 保存文件
df = pd.DataFrame({'top_k': range(1, 244), 'acc': top_k})
df['expected_rank'] = exp_rank   # 额外列
df.to_csv('top1_to_243.csv', index=False)
print('结果已写入 top1_to_243.csv')