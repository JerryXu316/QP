import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import gc
import os


# 1. 数据加载和预处理
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, sequence_length=5):
        # 读取数据
        data = pd.read_csv(file_path)

        # 输入特征和标签
        self.X = data.iloc[:, :5].values  # y(k), u(k-1), u(k-2), u(k-3), r(k)
        self.Y = data.iloc[:, 5:].values  # t(k+0) 到 t(k+4)

        # 标准化输入特征
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)  # 对输入特征进行标准化

        # 转换为 PyTorch tensor
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# 加载数据集
train_dataset = TimeSeriesDataset('solve_generated_data.csv')
valid_dataset = TimeSeriesDataset('solve_generated_data_e4.csv')
test_dataset = TimeSeriesDataset('solve_generated_data_e5.csv')

# 使用 DataLoader 批量加载数据
batch_size = 256  # 调整批量大小，确保内存利用率最大化
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 2. 神经网络模型（增加深度与更多非线性激活）
class DNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=512, output_size=5):
        super(DNN, self).__init__()

        # 增加层数和深度
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.5)  # Dropout with 30% probability
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x


# 初始化模型
model = DNN()

# 3. 检查是否有GPU可用，并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型移动到 GPU 或 CPU
model.to(device)


# 验证函数
def validate(model, valid_loader):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到 GPU 或 CPU

            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            running_loss += loss.item()

            # 计算准确率
            _, predicted_classes = torch.max(outputs, 1)  # 获取最大值的索引作为预测类别
            _, true_classes = torch.max(targets, 1)  # 获取真实类别索引

            correct_predictions += (predicted_classes == true_classes).sum().item()
            total_samples += true_classes.size(0)

    avg_loss = running_loss / len(valid_loader)
    accuracy = correct_predictions / total_samples * 100
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

    # 强制清理内存
    gc.collect()

# 5. 测试模型并计算准确率
def test(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)  # 将数据移到 GPU 或 CPU
            output = model(inputs)
            predictions.append(output)
            targets.append(target)

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    # 计算测试误差
    mse = nn.MSELoss()(predictions, targets)
    print(f"Test MSE: {mse.item()}")

    # 计算准确率
    _, predicted_classes = torch.max(predictions, 1)  # 获取最大值的索引作为预测类别
    predicted_classes = predicted_classes.cpu().numpy()  # 转换为 CPU 和 numpy 数组
    true_classes = targets.argmax(dim=1).cpu().numpy()  # 获取真实类别索引

    accuracy = (predicted_classes == true_classes).mean()  # 计算准确率
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 强制清理内存
    gc.collect()


# 4. 训练和验证（灵活设置训练轮数）
def train(model, train_loader, valid_loader, epochs=50, lr=1e-3, max_epochs=10000, log_dir='log', model_dir='model'):
    # 创建文件夹（如果不存在）
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 使用 ReduceLROnPlateau 来根据验证集损失动态调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.5)

    # 日志文件
    log_file = os.path.join(log_dir, 'training_log.txt')

    current_epoch = 0
    while current_epoch < epochs:
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到 GPU 或 CPU

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{current_epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

        # 每一轮结束后验证模型并输出准确率
        validate(model, valid_loader)

        # 更新学习率
        scheduler.step(running_loss / len(train_loader))  # Step with training loss

        # 保存日志文件
        with open(log_file, 'a') as f:
            f.write(f"Epoch [{current_epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}\n")

        # 每一百轮保存一次模型，并在test上测试准确率
        if (current_epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_{current_epoch + 1}.pth'))
            print(f'Model saved at epoch {current_epoch + 1}')
            # 在test集上测试
            test(model, test_loader)

        # 当学习率降低时，增加训练轮数
        if current_epoch >= max_epochs:
            break

        current_epoch += 1
        epochs += 5  # 每次学习率降低时增加5轮训练

        # 强制清理内存
        gc.collect()


# 训练模型
train(model, train_loader, valid_loader, epochs=50, lr=1e-3)





# 测试模型
test(model, test_loader)
