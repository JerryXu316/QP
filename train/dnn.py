# train_discrete.py
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. 数据集封装
# --------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.X = data.iloc[:, :5].values.astype('float32')
        self.Y = data.iloc[:, 5:].values.astype('float32')  # 原始就是 {-1,0,1}
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# --------------------------------------------------
# 2. 网络结构（末尾无 Tanh）
# --------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch: int, dropout: float):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(ch, ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ch, ch),
            nn.SiLU()
        )

    def forward(self, x):
        return x + self.f(x)


class NonlinearDecisionNet(nn.Module):
    def __init__(self,
                 in_dim: int = 5,
                 out_dim: int = 5,
                 base_ch: int = 512,
                 depth: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        assert depth >= 3
        layers = [nn.Linear(in_dim, base_ch), nn.SiLU(), nn.Dropout(dropout)]

        ch = base_ch
        for i in range(2, depth):
            next_ch = min(int(ch * 1.6), 2048)
            layers += [nn.Linear(ch, next_ch), nn.SiLU(), nn.Dropout(dropout)]
            if i % 2 == 0 and ch == next_ch:
                layers += [ResBlock(ch, dropout)]
            ch = next_ch

        layers += [nn.Linear(ch, out_dim)]  # 不经过 Tanh，直接输出实数
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------
# 3. 离散化 & 行准确率
# --------------------------------------------------
def discretize(x: torch.Tensor) -> torch.Tensor:
    """x: (B,5) → (B,5) 取整到 {-1,0,1}"""
    return torch.clip(torch.round(x), -1, 1)


def row_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_d = discretize(pred)
    return (pred_d == target).all(dim=1).float().mean().item()


# --------------------------------------------------
# 4. 通用 epoch：返回 (loss, row_acc)
# --------------------------------------------------
def run_epoch(model, loader, optimizer=None):
    device = next(model.parameters()).device
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_acc, n = 0.0, 0.0, 0
    criterion_mse = nn.MSELoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            out = model(x)
            loss = criterion_mse(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += row_accuracy(out, y) * x.size(0)
        n += x.size(0)

    return total_loss / n, total_acc / n


# --------------------------------------------------
# 5. 训练主循环
# --------------------------------------------------
def train(model, train_loader, valid_loader, epochs=100, lr=1e-3, model_dir='ckpt'):
    os.makedirs(model_dir, exist_ok=True)
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, valid_loader)
        scheduler.step(val_loss)
        print(f'Epoch {epoch:3d}/{epochs} | '
              f'train loss {tr_loss:.6f}  train acc {tr_acc*100:.2f}% | '
              f'val loss {val_loss:.6f}  val acc {val_acc*100:.2f}% | '
              f'lr {optimizer.param_groups[0]["lr"]:.2e}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'{model_dir}/best.pth')
        gc.collect()


# --------------------------------------------------
# 6. 主入口
# --------------------------------------------------
if __name__ == '__main__':
    # 1. 数据（确保三个 csv 文件存在）
    train_ds = TimeSeriesDataset('solve_generated_data.csv')
    valid_ds = TimeSeriesDataset('solve_generated_data_e4.csv')
    test_ds  = TimeSeriesDataset('solve_generated_data_e5.csv')

    batch = 256
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False)

    # 2. 模型 & 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NonlinearDecisionNet(depth=10, base_ch=512, dropout=0.3).to(device)

    # 3. 训练
    train(model, train_loader, valid_loader, epochs=200, lr=1e-3)

    # 4. 测试
    model.load_state_dict(torch.load('ckpt/best.pth'))
    test_loss, test_acc = run_epoch(model, test_loader)
    print(f'Test MSE: {test_loss:.6f}  Test Row Acc: {test_acc*100:.2f}%')