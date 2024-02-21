import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import floor

# 创建一个合成数据集
def generate_data(num_samples, num_features):
    x = np.random.uniform(-1, 1, (num_samples, num_features[0]-1))
    y = np.sum(x**2, axis=1)  # Sphere 函数
    x = np.concatenate([x, np.ones((num_samples, 1))],axis=1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 定义神经网络模型
class SimpleMLP(nn.Module):
    def __init__(self, num_features):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)
class LVMLP(nn.Module):
    def __init__(self, d_model, activation):
        super(LVMLP, self).__init__()
        self.d_model = d_model[:]
        self.d_model [0] = self.d_model[0]-1  # remove pid from first projection
        self.layers = nn.ModuleList(nn.Linear(self.d_model[i], self.d_model[i + 1]) for i in range(len(self.d_model) - 1))
        self.fixed_neuron = nn.Linear(1, self.d_model[1], bias=False)  # share bias?

        self.ln_layers = [i * 2 + 1 for i in range(floor(len(self.layers) / 2))]
        self.ln = nn.ModuleList(nn.LayerNorm(self.d_model[i]) for i in self.ln_layers)
        self.act = activation()
        self.model = nn.Sequential()

    def initialize(self, n_ln):

        self.model = nn.Sequential()
        for i in range(len(self.layers)):

            self.model.append(self.layers[i])

            if i + 1 in self.ln_layers and (i + 1) / 2 < n_ln:
                self.model.append(self.ln[int(i / 2)])
            self.model.append(self.act)
            # if i != len(self.layers) - 1:
            # else:
            #     self.model.append(self.layers[i])

    def forward(self, x):
        length = len(self.model)
        x = self.model[0](x[:, :-1]) + self.fixed_neuron(x[:, -1:])
        for i in range(1, length):
            x = self.model[i](x)
        return x
# 参数设置
num_samples = 1000
num_features = [20, 256, 512, 256, 1]  # 输入特征维度设置为 20
epochs = 10000
learning_rate = 0.001

# 数据生成
x_train, y_train = generate_data(num_samples, num_features)

# 模型初始化
model = LVMLP(num_features,torch.nn.GELU)
model.initialize(1)
criterion_p1 = nn.MSELoss()
optimizer_p1 = optim.Adam(model.model[1:].parameters(), lr=learning_rate)
optimizer_p2 = optim.Adam(model.model[:1].parameters(), lr=learning_rate)


# 训练模型
for epoch in range(epochs):

    optimizer_p1.zero_grad()
    outputs = model(x_train)
    loss_p1 = criterion_p1(outputs, y_train.view(-1, 1))
    loss_p1.backward()
    optimizer_p1.step()

    optimizer_p2.zero_grad()
    outputs = model(x_train)
    loss_p2 = outputs.sum()
    loss_p2.backward()
    optimizer_p2.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss_p1: {loss_p1.item():.4f}, Loss_p2: {loss_p2.item():.4f}')

# 测试模型
x_test, y_test = generate_data(100, num_features)
y_pred = model(x_test)