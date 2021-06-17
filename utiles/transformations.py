import torchvision


class TransformsSimCLR:
    """
    一种随机数据扩充模块，它对任意给定的数据实例进行随机转换，
    得到同一实例的两个相关视图，
    记为x̃i和x̃j，我们认为这是一个正对。
    """

    def __init__(self, size, train=True):
        """
        :param size:图片尺寸
        """
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.train = train

    def __call__(self, x):
        """

        :param x: 图片
        :return: x̃i和x̃j
        """

        if self.train:
            return self.train_transform(x), self.train_transform(x)
        else:
            return self.test_transform(x)
