# coding=UTF-8
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from utiles.transformations import TransformsSimCLR
from models.resnet50 import ResNet50
from models.resnet18 import ResNet18
from models.mlp import MLP
from utiles.loss_function import loss_function
import torch
import time


if __name__ == '__main__':
    image_size = 56  # 图片尺寸
    batch_size = 512  # 批次大小
    num_epochs = 100  # 要训练的迭代次数
    learn_rate = 0.001  # 学习率
    tau = 0.99  # 目标网络更新系数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 选择训练设备

    """实例化数据集和数据集加载器"""
    train_dataset = CIFAR10(
        root='dataset',
        train=True,
        transform=TransformsSimCLR(size=image_size),
        download=True
    )  # 训练数据集
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
    )  # 训练数据加载器

    online_net = ResNet18().to(device)  # 实例化online（在线网络）
    target_net = ResNet18().to(device)  # 实例化target（目标网络）

    """实例化prediction（预测网络）"""
    prediction = MLP(in_features=1000, hidden_features=2048, projection_features=1000).to(device)

    """实例化优化器，放入在线网络（online_net），和预测网络（prediction）的参数优化；目标网络（target_net）采用其他方式更新参数"""
    optimizer = torch.optim.Adam([{'params': online_net.parameters()}, {'params': prediction.parameters()}], lr=learn_rate)

    """训练train"""
    for epoch in range(num_epochs):
        for step, ((x_i, x_j), label) in enumerate(train_loader):
            """加载数据至GPU"""
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            """计算在线网络和目标网络的输出，同时对目标网络不更新梯度"""
            online_projection_one = online_net(x_i)
            with torch.no_grad():
                target_projection_one = target_net(x_j)
            loss_one = loss_function(prediction(online_projection_one), target_projection_one.detach())

            online_projection_two = online_net(x_j)  # 交换x_i与x_j，再计算损失；此步是为了高效利用数据，也可以不用
            with torch.no_grad():
                target_projection_two = target_net(x_i)
            loss_two = loss_function(prediction(online_projection_two), target_projection_two.detach())

            loss = (loss_one + loss_two).mean()  # 合计计算损失

            """update online parameters（更新在线网络的参数）"""
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 优化在线网络参数

            if step % 5 == 0:  # 打印训练中的情况
                print(f"Epoch [{epoch}/{num_epochs}]; Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")

            """
            update target parameters（更新目标网络的参数）
            target_parameter <=== target_parameter * beta + (1 - beta) * online_parameter
            """
            for target_parameter, online_parameter in zip(target_net.parameters(), online_net.parameters()):
                old_weight = target_parameter.data
                update = online_parameter.data
                target_parameter.data = old_weight * tau + (1 - tau) * update
            time.sleep(0.1)  # 训练太快，防止显卡过热，掉驱动
        """save net weights"""
        torch.save(online_net.state_dict(), 'net.pt')
