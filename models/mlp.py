from torch import nn


class MLP(nn.Module):
    """
    预测网络, 将在在线网络的输出投射至另一空间来预测目标网络的输出
    """
    def __init__(self, in_features, hidden_features, projection_features):
        """
        预测网络
        :param in_features: 输入特征数
        :param hidden_features: 隐藏特征数
        :param projection_features: 投影特征数
        """
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, projection_features),
        )

    def forward(self, x):
        return self.layer(x)
