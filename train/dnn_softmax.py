import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---------- 0. 额外 import ----------
import torch.nn.functional as F   # 新增

# ---------- 1. 数据集 ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.X = data.iloc[:, :5].values.astype('float32')
        # 关键：标签改为 int64 的 0/1/2
        self.Y = data.iloc[:, 5:].values.astype('int64') + 1   # {-1,0,1} -> {0,1,2}
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)        # (N,5) 每列是 0/1/2 类别索引

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ---------- 2. 网络 ----------

class ResBlock(nn.Module):
    def __init__(self, ch: int, dropout: float):
        super().__init__()
        self.f = nn.Sequential(
            nn.LayerNorm(ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
            nn.Dropout(dropout),
            nn.LayerNorm(ch),
            nn.SiLU(),
            nn.Linear(ch, ch)
        )

    def forward(self, x):
        return x + self.f(x)


class NonlinearDecisionNet(nn.Module):
    def __init__(self,
                 in_dim=5,
                 base_ch=512,
                 depth=15,
                 dropout=0.2):
        super().__init__()
        assert depth >= 3
        layers = [nn.Linear(in_dim, base_ch),
                  nn.LayerNorm(base_ch), nn.SiLU()]

        ch = base_ch
        for i in range(2, depth):
            if (i - 2) % 3 == 0:
                layers.append(ResBlock(ch, dropout))
            next_ch = min(int(base_ch * (1 + 0.5 * abs((i - 2) % 6 - 3))), 1536)
            if ch != next_ch:
                layers += [nn.Linear(ch, next_ch),
                           nn.LayerNorm(next_ch), nn.SiLU()]
                ch = next_ch

        # 输出 5×3，后面 reshape 成 (N,5,3)
        self.net = nn.Sequential(*layers,
                                 nn.Linear(ch, 5 * 3))

    def forward(self, x):
        out = self.net(x)              # (N,15)
        return out.view(-1, 5, 3)      # (N,5,3)

# ---------- 3. 评估指标 ----------
def row_metrics(logits, y_true):
    probs = F.softmax(logits, dim=-1)        # (N,5,3)
    joint = torch.ones(probs.shape[0], 243, device=probs.device)
    idx = 0
    for t0, t1, t2, t3, t4 in torch.cartesian_prod(
            torch.arange(3), torch.arange(3),
            torch.arange(3), torch.arange(3),
            torch.arange(3)):
        p = probs[:, 0, t0] * probs[:, 1, t1] * probs[:, 2, t2] * probs[:, 3, t3] * probs[:, 4, t4]
        joint[:, idx] = p
        idx += 1

    y_true_idx = (
        y_true[:, 0] * 3**4 +
        y_true[:, 1] * 3**3 +
        y_true[:, 2] * 3**2 +
        y_true[:, 3] * 3**1 +
        y_true[:, 4] * 3**0
    )

    order = torch.argsort(joint, dim=1, descending=True)
    ranks = (order == y_true_idx.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    top1 = (ranks == 1).float().mean()
    top3 = (ranks <= 3).float().mean()
    exp_rank = ranks.float().mean()
    return top1, top3, exp_rank

# ---------- 4. 通用 epoch ----------
# ---------- 在 run_epoch 里 ----------
def run_epoch(model, loader, optimizer=None, rank_lambda=0.03):
    device = next(model.parameters()).device
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n = 0.0, 0
    all_logits, all_y = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)                       # (N,5,3)
            # 5 列交叉熵
            ce_loss = 0.0
            for j in range(5):
                ce_loss += F.cross_entropy(logits[:, j, :], y[:, j])
            ce_loss /= 5

            # ---------------- 期望排名惩罚 ----------------
            # 先算联合概率与排名
            probs = F.softmax(logits, dim=-1)
            joint = torch.ones(probs.shape[0], 243, device=device)
            idx = 0
            for t0, t1, t2, t3, t4 in torch.cartesian_prod(
                    *[torch.arange(3)] * 5):
                p = probs[:, 0, t0] * probs[:, 1, t1] * probs[:, 2, t2] * probs[:, 3, t3] * probs[:, 4, t4]
                joint[:, idx] = p
                idx += 1

            y_true_idx = (
                y[:, 0] * 3**4 + y[:, 1] * 3**3 +
                y[:, 2] * 3**2 + y[:, 3] * 3**1 + y[:, 4] * 3**0
            )
            order = torch.argsort(joint, dim=1, descending=True)
            ranks = (order == y_true_idx.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
            rank_penalty = (ranks - 1).clamp(min=0).float().mean()

            loss = ce_loss + rank_lambda * rank_penalty
            # -------------------------------------------------

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_logits.append(logits.detach())
        all_y.append(y)

    # 评估用 ce_loss 不含惩罚，方便对比
    all_logits = torch.cat(all_logits)
    all_y = torch.cat(all_y)
    top1, top3, exp_rank = row_metrics(all_logits, all_y)
    return total_loss / n, top1, top3, exp_rank


# ---------- 5. 训练 ----------
def train(model, train_loader, valid_loader, epochs=3, lr=1e-4, save_dir='model_dnn'):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        tr_loss, tr_top1, tr_top3, tr_rank = run_epoch(model, train_loader, optimizer)
        val_loss, val_top1, val_top3, val_rank = run_epoch(model, valid_loader)
        scheduler.step(val_loss)

        print(f'Epoch {epoch:3d}/{epochs} | '
              f'train loss {tr_loss:.6f} | train Top-1 {tr_top1*100:.2f}% | '
              f'val Top-1 {val_top1*100:.2f}% | val Top-3 {val_top3*100:.2f}% | '
              f'val Rank {val_rank:.2f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_5x3.pth'))

# ---------- 6. 主入口 ----------
if __name__ == '__main__':
    train_ds = TimeSeriesDataset('solve_generated_data.csv')
    valid_ds = TimeSeriesDataset('solve_generated_data_e4.csv')
    test_ds  = TimeSeriesDataset('solve_generated_data_e5.csv')

    batch = 256
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NonlinearDecisionNet(depth=15, base_ch=512, dropout=0.3).to(device)

    train(model, train_loader, valid_loader, epochs=100, lr=1e-4)

    # 测试
    model.load_state_dict(torch.load('model_dnn/best_5x3.pth'))
    test_loss, test_top1, test_top3, test_rank = run_epoch(model, test_loader)
    print(f'Test Loss {test_loss:.6f} | '
          f'Test Top-1 {test_top1*100:.2f}% | '
          f'Test Top-3 {test_top3*100:.2f}% | '
          f'Test Rank {test_rank:.2f}')