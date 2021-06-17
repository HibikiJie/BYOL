from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from utiles.transformations import TransformsSimCLR
from models.resnet18 import ResNet18
from torch import nn
import torch

batch_size = 128  # 批次
num_epochs = 100  # 总的迭代次数
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # 选择设备
"""实例化数据集和数据集加载器"""
train_dataset = CIFAR10(
    root='dataset',  # 路径
    train=True,  # 训练模型
    transform=TransformsSimCLR(size=56),  # 图片转换器
    download=True  # 下载文件
)  # 训练数据集
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    drop_last=True,
    num_workers=4,
)  # 训练数据加载器
test_dataset = CIFAR10(
    root='dataset',
    train=False,
    transform=TransformsSimCLR(size=56, train=False),
    download=True
)  # 测试数据集
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    drop_last=False,
    num_workers=4,
)  # 测试数据加载器

net = ResNet18()  # 实例化残差18网络
net.load_state_dict(torch.load('net.pt',map_location='cpu'))  # 加载预训练权重
net.fc = nn.Linear(512, 10)  # 重新赋值全连接线性分类层
net = net.to(device)  # 网络加载进训练设备
optimizer = torch.optim.Adam(net.fc.parameters())  # 将网络线性分类层放入优化器中，只优化线性分类层
loss_function = nn.CrossEntropyLoss()  # 实例化，交叉熵损失函数

"""训练线性分类层"""
for epoch in range(num_epochs):
    '''train'''
    loss_sum = 0
    # net.load_state_dict(torch.load('net.pt'))
    for step, ((x_i, x_j), label) in enumerate(train_loader):
        x_i = x_i.to(device)
        label = label.to(device)

        # with torch.no_grad():
            # projection = net(x_i)
        # predict = linear(projection.detach())
        predict = net(x_i)
        loss = loss_function(predict, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        # print(f"Epoch {epoch}; Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")

    '''val'''
    """测试，线性整体效果"""
    with torch.no_grad():
        c_sum = 0
        y_true = []
        y_pred = []
        for step, (x, label) in enumerate(test_loader):
            x = x.to(device)
            label = label.to(device)

            # projection = net(x)  # 预测
            # predict = linear(projection.detach())
            predict = net(x)
            y_true.extend(label.cpu().numpy().tolist())
            out = torch.argmax(predict, dim=1)
            y_pred.extend(out.detach().cpu().numpy().tolist())
            c = out == label  # 判断预测值和标签是否相等
            c = c.sum()  # 计算预测正确的个数
            c_sum += c

        logs = f'{epoch}、loss: {loss_sum / len(train_loader)} 测试集正确率：{str((c_sum / len(test_dataset)).item() * 100)[:6]}%'
        print(logs)
