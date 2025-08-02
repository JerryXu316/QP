# train_discrete_final.py
import os, gc, torch, torch.nn as nn, torch.optim as optim, pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---------- 数据集 ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.X = data.iloc[:, :5].values.astype('float32')
        self.Y = data.iloc[:, 5:].values.astype('float32')  # 原始就是 {-1,0,1}
        self.scaler = StandardScaler().fit_transform(self.X)
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ---------- 残差块 ----------
class ResBlock(nn.Module):
    def __init__(self, ch, dropout):
        super().__init__()
        self.f = nn.Sequential(
            nn.LayerNorm(ch), nn.SiLU(),
            nn.Linear(ch, ch), nn.Dropout(dropout),
            nn.LayerNorm(ch), nn.SiLU(),
            nn.Linear(ch, ch)
        )
    def forward(self, x): return x + self.f(x) * 0.707  # √0.5 缩放

# ---------- Ultra 网络 ----------
class UltraDeepNet(nn.Module):
    def __init__(self,
                 in_dim=5,
                 out_dim=5,
                 base_ch=512,
                 max_ch=2048,
                 depth=25,
                 dropout=0.2):
        super().__init__()
        layers = [nn.Linear(in_dim, base_ch), nn.LayerNorm(base_ch), nn.SiLU()]
        ch = base_ch
        # 线性通道伸缩
        for i in range(2, depth):
            next_ch = int(base_ch + (max_ch - base_ch) * (i - 1) / (depth - 2))
            next_ch = min(next_ch, max_ch)
            layers.append(ResBlock(ch, dropout))
            if ch != next_ch:
                layers += [nn.Linear(ch, next_ch), nn.LayerNorm(next_ch), nn.SiLU()]
                ch = next_ch
        layers.append(nn.Linear(ch, out_dim))
        self.net = nn.Sequential(*layers)
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)

# ---------- 离散化 ----------
discretize   = lambda x: torch.clip(torch.round(x), -1, 1)

# ---------- 行准确率 ----------
def row_accuracy(pred, target):
    """pred: (B,5) 连续 → 离散后比对整行"""
    return (discretize(pred) == target).all(dim=1).float().mean().item()

# ---------- epoch ----------
def run_epoch(model, loader, optimizer=None):
    device = next(model.parameters()).device
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_acc, n = 0.0, 0.0, 0
    criterion = nn.MSELoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += row_accuracy(out, y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_acc / n

# ---------- 训练 ----------
def train(model, train_loader, valid_loader, epochs=200, lr=2e-4, save_dir='model_dnn'):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    best = float('inf')
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, valid_loader)
        scheduler.step(val_loss)
        print(f'E{epoch:03d} | '
              f'TrainLoss {tr_loss:.5f} TrainAcc {tr_acc*100:6.2f}% | '
              f'ValLoss {val_loss:.5f} ValAcc {val_acc*100:6.2f}% | '
              f'lr {optimizer.param_groups[0]["lr"]:.1e}')
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_discrete.pth'))
        gc.collect()

# ---------- 主入口 ----------
if __name__ == '__main__':
    train_ds = TimeSeriesDataset('solve_generated_data.csv')
    valid_ds = TimeSeriesDataset('solve_generated_data_e4.csv')
    test_ds  = TimeSeriesDataset('solve_generated_data_e5.csv')

    batch = 256
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UltraDeepNet(depth=25, max_ch=2048, dropout=0.2).to(device)
    train(model, train_loader, valid_loader, epochs=200, lr=2e-4)

    model.load_state_dict(torch.load('model_dnn/best_discrete.pth'))
    test_loss, test_acc = run_epoch(model, test_loader)
    print(f'Test  MSE {test_loss:.6f}  Row-Acc {test_acc*100:.2f}%')