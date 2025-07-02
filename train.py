import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

class MIQPDataset(Dataset):
    def __init__(self, csv_path, N=10, Nc=10):
        self.df = pd.read_csv(csv_path)
        self.N = N
        self.Nc = Nc

        def parse_float_array(s):
            return np.array([float(x) for x in s.split(',')])

        self.df['z_pos_arr'] = self.df['z_pos'].apply(parse_float_array)
        self.df['z_neg_arr'] = self.df['z_neg'].apply(parse_float_array)
        self.df['label'] = self.df.apply(lambda row: np.concatenate([row['z_pos_arr'], row['z_neg_arr']]), axis=1)

        self.df.sort_values('t', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.y_arr = self.df['y'].values
        self.u_arr = self.df['u'].values
        self.labels = np.stack(self.df['label'].values)

        self.y_mean = self.y_arr.mean()
        self.y_std = self.y_arr.std()
        self.u_mean = self.u_arr.mean()
        self.u_std = self.u_arr.std()

        self.y_norm = (self.y_arr - self.y_mean) / self.y_std
        self.u_norm = (self.u_arr - self.u_mean) / self.u_std

    def __len__(self):
        return len(self.df) - self.N + 1

    def __getitem__(self, idx):
        y_seq = self.y_norm[idx:idx+self.N]
        u_seq = self.u_norm[idx:idx+self.N]
        x = np.concatenate([y_seq, u_seq])
        label = self.labels[idx + self.N - 1]
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
    csv_path = "miqp_data.csv"
    N = 10
    Nc = 10
    batch_size = 64
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MIQPDataset(csv_path, N=N, Nc=Nc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MIQPNet(input_dim=2*N, output_dim=2*Nc)

    train(model, dataloader, device, epochs=epochs, save_path="miqp_model.pt")

    # 评估训练集准确率
    evaluate(model, dataloader, device)
