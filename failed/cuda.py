import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

class MIQPDataset(Dataset):
    def __init__(self, csv_path, N=10, Nc=10):
        self.df = pd.read_csv(csv_path)
        self.N = N
        self.Nc = Nc

        # 提取 z_pos 和 z_neg 列
        z_pos_cols = [col for col in self.df.columns if col.startswith('z_pos_')]
        z_neg_cols = [col for col in self.df.columns if col.startswith('z_neg_')]

        # 将 z_pos 和 z_neg 列转换为数组
        self.df['z_pos_arr'] = self.df[z_pos_cols].values.tolist()
        self.df['z_neg_arr'] = self.df[z_neg_cols].values.tolist()
        self.df['label'] = self.df.apply(lambda row: np.concatenate([row['z_pos_arr'], row['z_neg_arr']]), axis=1)

        self.y_arr = self.df['y_t'].values
        self.u_arr = self.df['u_tm1'].values
        self.labels = np.stack(self.df['label'].values)

        self.y_mean = self.y_arr.mean()
        self.y_std = self.y_arr.std()
        self.u_mean = self.u_arr.mean()
        self.u_std = self.u_arr.std()

        self.y_norm = (self.y_arr - self.y_mean) / self.y_std
        self.u_norm = (self.u_arr - self.u_mean) / self.u_std

        # 计算有效的数据索引范围，避免跨越仿真边界
        self.valid_indices = [i for i in range(len(self.df) - self.N + 1) if self.df.iloc[i + self.N - 1]['t'] < 200]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 使用有效索引
        true_idx = self.valid_indices[idx]
        y_seq = self.y_norm[true_idx:true_idx + self.N]
        u_seq = self.u_norm[true_idx:true_idx + self.N]
        x = np.concatenate([y_seq, u_seq])
        label = self.labels[true_idx + self.N - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class MIQPNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MIQPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train(model, dataloader, device, epochs=20, lr=1e-3, save_path="miqp_model.pt"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def evaluate(model, dataloader, device):
    model.eval()
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == yb).sum().item()
            total_samples += yb.numel()
    acc = correct / total_samples
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

if __name__ == "__main__":
    csv_path = "miqp_sim_data.csv"
    N = 10
    Nc = 10
    batch_size = 64
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据集
    dataset = MIQPDataset(csv_path, N=N, Nc=Nc)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = MIQPNet(input_dim=2*N, output_dim=2*Nc)

    # 训练模型
    train(model, train_dataloader, device, epochs=epochs, save_path="miqp_model.pt")

    # 评估测试集准确率
    evaluate(model, test_dataloader, device)