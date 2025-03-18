import torch
# print(torch.__version__)
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 检测CUDA是否可用
use_cuda = torch.cuda.is_available()
# print(use_cuda)

# 设置device变量
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 设置对数据进行处理的逻辑
transform = transforms.Compose([
    # 让数据转成Tensor张量
    transforms.ToTensor(),
    # 让图片数据进行标准归一化，0.1307是标准归一化的均值，0.3081对应的是标准归一化的方差
    transforms.Normalize((0.1307,), (0.3081,))
])

# 读取数据
datasets1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
datasets2 = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 设置数据加载器，顺带手设置批次大小和是否打乱数据顺序
train_loader = torch.utils.data.DataLoader(datasets1, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets2, batch_size=1000)

# for batch_idx, data in enumerate(train_loader, 0):
#     inputs, targets = data
#     # view在下一行会把我们的训练集(60000,1,28,28)转换成(60000,28*28)
#     x = inputs.view(-1, 28*28)
#     # 计算所有训练样本的标准差和均值
#     x_std = x.std().item()
#     x_mean = x.mean().item()
#
# print('均值mean为：'+str(x_mean))
# print('标准差std为：'+str(x_std))


# 通过自定义类来构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# 创建一个模型实例
model = Net().to(device)


# 定义训练模型的逻辑
def train_step(data, target, model, optimizer):
    optimizer.zero_grad()
    output = model(data)
    # nll代表着 negative log likely hood 负对数似然
    loss = F.nll_loss(output, target)
    # 反向传播的本质是不是就是去求梯度
    loss.backward()
    # 本质就是应用梯度去调参
    optimizer.step()
    return loss


# 定义测试模型的逻辑
def test_step(data, target, model, test_loss, correct):
    output = model(data)
    # 累积的批次损失
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    # 获得对数概率最大值对应的索引号，这里其实就是类别号
    pred = output.argmax(dim=1, keepdims=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss, correct


# 创建训练调参使用的优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 真正的分轮次训练
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = train_step(data, target, model, optimizer)
        # 每隔10个批次，打印信息
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(epoch, batch_idx*len(data),
                  len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            test_loss, correct = test_step(data, target, model, test_loss, correct)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(test_loader.dataset)))
