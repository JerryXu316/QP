# train_15L.py
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---------- 1. 数据集 ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.X = data.iloc[:, :5].values.astype('float32')
        self.Y = data.iloc[:, 5:].values.astype('float32')
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ---------- 2. Pre-Activation ResBlock ----------
class ResBlock(nn.Module):
    def __init__(self, ch, dropout):
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

# ---------- 3. 15 层网络 ----------
class NonlinearDecisionNet(nn.Module):
    def __init__(self,
                 in_dim=5,
                 out_dim=5,
                 base_ch=512,
                 depth=15,
                 dropout=0.3):
        super().__init__()
        assert depth >= 3
        layers = [nn.Linear(in_dim, base_ch), nn.LayerNorm(base_ch), nn.SiLU()]

        ch = base_ch
        for i in range(2, depth):
            # 每 3 层做一次残差
            if (i - 2) % 3 == 0:
                layers.append(ResBlock(ch, dropout))
            # 通道瓶颈：512→768→1024→768→512
            next_ch = min(int(base_ch * (1 + 0.5 * abs((i - 2) % 6 - 3))), 1536)
            if ch != next_ch:
                layers += [nn.Linear(ch, next_ch), nn.LayerNorm(next_ch), nn.SiLU()]
                ch = next_ch

        layers.append(nn.Linear(ch, out_dim))
        self.net = nn.Sequential(*layers)

        # kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ---------- 4. 离散化 ----------
def discretize(x):
    return torch.clip(torch.round(x), -1, 1)

def row_accuracy(pred, target):
    return (discretize(pred) == target).all(dim=1).float().mean().item()

# ---------- 5. 通用 epoch ----------
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += row_accuracy(out, y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_acc / n

# ---------- 6. 训练 ----------
def train(model, train_loader, valid_loader, epochs=200, lr=1e-4, save_dir='model_dnn'):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

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
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_15L.pth'))
        gc.collect()

# ---------- 7. 主入口 ----------
if __name__ == '__main__':
    train_ds = TimeSeriesDataset('solve_generated_data.csv')
    valid_ds = TimeSeriesDataset('solve_generated_data_e4.csv')
    test_ds  = TimeSeriesDataset('solve_generated_data_e5.csv')

    batch = 256
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NonlinearDecisionNet(depth=15, base_ch=512, dropout=0.2).to(device)

    train(model, train_loader, valid_loader, epochs=100, lr=1e-4)

    model.load_state_dict(torch.load('model_dnn/best_15L.pth'))
    test_loss, test_acc = run_epoch(model, test_loader)
    print(f'Test MSE: {test_loss:.6f}  Test Row Acc: {test_acc*100:.2f}%')