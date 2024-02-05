import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 创建一个合成数据集
def generate_data(num_samples, num_features):
    x = np.random.uniform(-5.12, 5.12, (num_samples, num_features))
    y = np.sum(x**2, axis=1)  # Sphere 函数
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class SimpleMLP(nn.Module):
    def __init__(self, num_features):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 128),  # 增加神经元数量
            nn.ReLU(),
            nn.Linear(128, 256),           # 新增额外层
            nn.ReLU(),
            nn.Linear(256, 256),           # 新增额外层
            nn.ReLU(),
            nn.Linear(256, 128),           # 新增额外层
            nn.ReLU(),
            nn.Linear(128, 1)              # 输出层
        )

    def forward(self, x):
        return self.layers(x)


# 参数设置
num_samples = 500
num_features = 5 # 输入特征维度设置为 20
epochs = 500
learning_rate = 0.001
batch_size = 32  # 定义批大小

# 数据生成
x_train, y_train = generate_data(num_samples, num_features)

# 创建数据加载器
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 模型初始化
model = SimpleMLP(num_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型并输出每个测试点的标签值、预测值和误差
x_test, y_test = generate_data(100, num_features)
y_pred = model(x_test)

# 计算和输出每个测试点的标签值、预测值和误差
for i in range(len(y_test)):
    label = y_test[i].item()
    prediction = y_pred[i].item()
    error = abs(label - prediction)
    print(f'Test Point {i+1}, Label: {label:.4f}, Prediction: {prediction:.4f}, Error: {error:.4f}')
