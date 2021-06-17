import torch


def loss_function(predict, target):
    """
    损失函数，比较余弦相似度。归一化的欧氏距离等价于余弦相似度
    :param predict: online net输出的prediction
    :param target: target网络输出的projection
    :return: loss(损失)
    """
    return 2-2*torch.cosine_similarity(predict, target, dim=-1)