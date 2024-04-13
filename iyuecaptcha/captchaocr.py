import torch
import os.path
import platform
import torch.nn as nn
from PIL import Image
from loguru import logger
from torchvision import transforms
import torchvision.models as models
from iyuecaptcha.tools import Tools
from torch.utils.data import Dataset, DataLoader
from iyuecaptcha.tools import get_defualt_device
from torch.utils.tensorboard import SummaryWriter

tool = Tools()


class CaptchaOCRDataset(Dataset):
    def __init__(self, root_dir, split):
        '''
        验证码图片数据加载

        :param root_dir: 需要加载的图片目录路径
        :param split: 路径标签分割 从文件0A7NC-1713030561.png 获取标签 0A7NC
        '''
        super(CaptchaOCRDataset, self).__init__()
        self.__split = split
        # 加载所有图片路径
        self.__list_image_path = []
        skip_count = 0
        for path in os.listdir(root_dir):
            if self.__split not in path:
                logger.debug(f'{path}: Not meeting the requirements,skip')
                skip_count = skip_count + 1
                continue
            self.__list_image_path.append(os.path.join(root_dir, path))
        # self.__list_image_path = [os.path.join(root_dir, path) for path in os.listdir(root_dir)]
        self.__transform = transforms.Compose([
            transforms.Resize((tool.height, tool.width)),
            transforms.ToTensor(),
            transforms.Grayscale()
        ])
        logger.info(f'{len(self.__list_image_path)} images loaded, {skip_count} images skip')

    def __getitem__(self, item):
        """
        根据索引获取图像及其对应的验证码。

        参数:
        - self: 对象自身。
        - item: 索引值，用于从图像路径列表中获取特定的图像路径。

        返回值:
        - image_transform: 经过变换处理的图像。
        - captcha: 图像对应的验证码文本。
        """
        # 根据索引获取图像路径
        image_path = self.__list_image_path[item]
        # 打开图像
        image = Image.open(image_path)
        # 对图像进行预定义的变换处理
        image_transform = self.__transform(image)
        # 根据操作系统不同选择路径分隔符
        split = '/'
        if platform.system() == 'Windows':
            split = '\\'
        # 从图像路径中提取文件名，并进一步提取验证码文本
        image_name = self.__list_image_path[item].split(split)[-1]
        captcha = image_name.split(self.__split)[0]
        # 将验证码文本转换为向量表示
        '''
        .view(1, -1): 
            这是PyTorch中的view()函数调用，作用是对上一步得到的向量或矩阵进行维度调整。这里的参数(1, -1)指示了新的形状：
        1: 表示希望保持一个维度的大小为1。在很多深度学习应用中，尤其是与序列数据相关的模型，可能会要求输入数据具有 batch dimension（批量维度）。
            这里设置为1意味着即使原始验证码向量只有一个样本，也要将其包装在一个“批量”中，使得模型可以接受这样的输入。
        -1: 在PyTorch中，-1作为一个特殊的值，表示该维度的大小将在运行时自动计算，以确保总体积（元素总数）保持不变。它允许用户在不知道确切值的情况下调整其他维度，只要总体积匹配即可。
            在这里，-1意味着将剩余的所有元素都分配到这个维度上，形成一个列向量。
        总体来说，.view(1, -1)的作用是将__text2vec()返回的结果转换为形状为 (1, N) 的张量，其中 N 是原始向量的长度，确保了数据满足模型所需的输入格式（单样本、一列向量）。
        [0]: 最后，通过索引操作取出重塑后的张量的第一个（也是唯一一个）元素。由于已经通过view(1, -1)确保了数据是一个批量大小为1的张量，
            这里的操作实际上只是去除掉了多余的批量维度，直接返回经过编码和重塑的单个验证码向量。
        '''
        captcha = self.__text2vec(captcha).view(1, -1)[0]  # tools.explain.view
        return image_transform, captcha

    def __len__(self):
        return self.__list_image_path.__len__()

    def __text2vec(self, text):
        """
        将文本转换为向量表示。

        参数:
        text: str, 输入的文本字符串，预期为验证码等短字符串。

        返回值:
        vecs: torch.Tensor, 转换后的向量表示，其中vecs的shape为(captcha_length, captcha_array_length)，
              每个位置上的值表示对应字符在该文本中是否出现，出现则为1，不出现则为0。
        """
        # 初始化一个全零的向量，其大小根据验证码长度和字符集大小确定
        vecs = torch.zeros((tool.captcha_lenth, tool.captcha_array.__len__()))
        for i in range(len(text)):
            # 对于文本中的每个字符，将其在向量中的位置设置为1
            vecs[i][tool.captcha_array.index(text[i])] = 1
        return vecs

    @staticmethod
    def vec2text(vec):
        '''
        将向量转换为文本表示。

        :param vec:
        :return:
        '''
        # 获取向量中每个位置上值为1的索引，即为字符在字符集的位置
        '''
        vec =   [[2, 5,  3],
                [6, 1, 7, 4],
                [9, 8, 6, 2]]
       执行 vec=torch.argmax(vec, dim=1) 后，得到的新张量将是：
       result = [1, 2, 0] 
        result[0]=1（索引为0）的最大值位于第1列（索引为1），即 5；
        result[1]=2 表示第二行最大值在第2列（值为 7），
        result[2]=0 表示第三行最大值在第0列（值为 9）。
        '''
        vecs = torch.argmax(vec, dim=1)
        return ''.join([tool.captcha_array[i] for i in vecs])


class CaptchaOCRModel(nn.Module):
    """
    CaptchaOCRModel 类用于定义一个针对验证码识别的卷积神经网络模型。

    该模型基于ResNet50架构进行改造，以适应验证码图像的识别任务。
    """

    def __init__(self):
        """
        初始化CaptchaOCRModel实例。

        """
        super(CaptchaOCRModel, self).__init__()
        # 初始化ResNet50模型，不加载预训练权重
        self.__model = models.resnet50(weights=None)
        # 修改输入层 修改ResNet50的输入层以适应单通道输入。 直接打印 model 查看原始信息
        self.__model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 修改输出层 修改最后一层全连接层，以输出与验证码长度和字符集大小相应的类别数。
        self.__model.fc = nn.Linear(in_features=2048, out_features=tool.captcha_lenth * tool.captcha_array.__len__())

    def forward(self, x):
        """
        定义模型的前向传播路径。

        参数:
        - x : 输入图像的张量

        返回值:
        - x : 经过模型处理后的输出张量
        """
        x = self.__model(x)  # 调用模型进行前向传播
        return x


class CaptchaOCR:
    def __init__(self, captcha_lenth, captcha_array, width, height):
        """
        初始化CaptchaOCR实例。

        :param captcha_lenth: 验证码长度
        :param captcha_array: 验证码包含的字符
        :param width: 验证码图片的宽度
        :param height: 验证码图片的高度
        """
        # self.__model = CaptchaOCRModel
        self.__model_pth_path = None
        # 设定训练设备，默认使用GPU
        self.__device = get_defualt_device()
        # 初始化数据
        global tool
        tool.captcha_lenth = captcha_lenth
        tool.captcha_array = captcha_array
        tool.width = width
        tool.height = height

    def train(self, train_path, model_pth_path='model.pth', train_path_split='-', epochs=10, log_dir="log_dir"):
        """
        训练验证码识别模型。

        :param train_path: 训练数据集的路径。该路径指向包含用于训练模型的验证码图像及对应标签的数据集。
        :param model_pth_path: data/model.pth 训练完成后模型保存的路径。模型训练结束后，将保存的模型状态字典（state_dict）写入此路径指定的文件中。
        :param train_path_split: 路径标签分割 从文件0A7NC-1713030561.png 获取标签 0A7NC
        :param epochs: 训练的轮数，默认为10轮。模型将在整个训练数据集上迭代此数量的次数以完成训练过程。
        :param log_dir: tensorboard 日志目录
        """

        if os.path.exists(model_pth_path):
            logger.debug('model.pth is exists')
            return

        # 加载训练数据集
        train_datas = CaptchaOCRDataset(train_path, train_path_split)
        if train_datas.__len__() == 0:
            raise Exception('load train data lenth is 0')
        # 创建数据加载器 batch_size 每批次加载多少个样本（默认值：1）。
        train_dataloader = DataLoader(train_datas, batch_size=64, shuffle=True)
        # 加载模型到指定设备
        model = CaptchaOCRModel().to(self.__device)

        # 定义损失函数
        loss_fn = nn.MultiLabelSoftMarginLoss().to(self.__device)
        # 定义优化器：使用Adam算法，学习率为0.001
        # lr (learning rate) 表示学习率，控制每次参数更新的幅度。此处设置为0.001，
        # 即每次梯度更新时，模型参数将以0.001的比例沿梯度方向进行调整。
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # 创建TensorBoard SummaryWriter实例
        writer = SummaryWriter(log_dir=log_dir)

        # 记录总步数
        total_step = 0
        # 开始训练
        for _ in range(epochs):
            for i, (image, label) in enumerate(train_dataloader):
                # 将数据移动到指定设备
                image = image.to(self.__device)
                label = label.to(self.__device)
                # 前向传播
                outputs = model(image)
                # 计算损失
                loss = loss_fn(outputs, label)
                # 清空梯度
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()
                # 每10步打印一次训练状态
                if i % 100 == 0:
                    total_step += 1
                    logger.info(f'train total:{total_step * 100}, loss:{loss.item()}')
                    writer.add_scalar('Loss train', loss.item(), total_step)

        writer.close()
        logger.info('Finished Training')

        # 保存训练好的模型
        torch.save(model.state_dict(), model_pth_path)
        self.__model_pth_path = model_pth_path
        logger.info(f'model save to {model_pth_path}')

    def predicts(self, images_path, model_pth_path='model.pth', train_path_split='-'):
        '''
        通过模型进行批量预测，输出预测结果

        :param images_path: 需要预测的图像目录路径
        :param model_pth_path: 使用的模型路径 如果为空使用上一次训练的模型
        :param train_path_split: 路径标签分割 从文件0A7NC-1713030561.png 获取标签 0A7NC
        :return:
        '''
        if model_pth_path is None:
            if self.__model_pth_path is None:
                raise FileNotFoundError('model_pth_path is None')
            model_pth_path = self.__model_pth_path

        if not os.path.exists(images_path):
            raise FileNotFoundError(f'images_path:{images_path} not found')

        # 加载需要预测的数据
        datas = CaptchaOCRDataset(images_path, train_path_split)
        # batch_size 每批次加载多少个样本（默认值：1）。
        dataloader = DataLoader(datas, shuffle=False)
        if datas.__len__() == 0:
            raise Exception('no images')

        # 加载模型到指定设备
        model = CaptchaOCRModel()
        model.load_state_dict(torch.load(model_pth_path))
        model.to(self.__device)
        model.eval()

        success_count = 0
        for i, (image, label) in enumerate(dataloader):
            # 将数据移动到指定设备
            img = image.to(self.__device)
            label = label.to(self.__device)

            label_vivew = label.view(-1, tool.captcha_array.__len__())
            lables_text = CaptchaOCRDataset.vec2text(label_vivew)

            predict_outputs = model(img).view(-1, tool.captcha_array.__len__())
            predict_text = CaptchaOCRDataset.vec2text(predict_outputs)

            if lables_text == predict_text:
                success_count = success_count + 1
                logger.success(f'{i}:{lables_text}=={predict_text}')
            else:
                logger.error(f'{i}:{lables_text}!={predict_text}')

        logger.info(f'Accuracy:{success_count / datas.__len__() * 100}')

    def predict(self, image_path, model_pth_path='model.pth'):
        """
        通过模型进行预测，输出预测结果

        :param image_path: 需要预测的图像路径
        :param model_pth_path: 使用的模型路径 如果为空使用上一次训练的模型
        """
        # 判断并设置模型路径
        if model_pth_path is None:
            if self.__model_pth_path is None:
                raise FileNotFoundError('model_pth_path is None')
            model_pth_path = self.__model_pth_path

        # 检查图像路径是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'image_path:{image_path} not found')

        # 加载和预处理图像
        image = Image.open(image_path)
        tersor_img = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((tool.height, tool.width)),
            transforms.ToTensor()
        ])
        img = tersor_img(image).to(self.__device)
        '''
        -1 表示这一维度的大小将根据其他已知维度和原始张量的元素总数自动计算得出
        1 表示在新形状中创建一个单通道维度,将输入视为一个只有一个颜色通道（例如灰度图像）的数据
        tools.height 和 tools.width 是表示高度和宽度的变量值，
        它们指定了重塑后张量在相应维度上的像素数。这两个值应该是事先定义好的整数，代表期望的图像高度和宽度。
        '''
        # 重塑图像张量以适配模型输入
        reshape_img = torch.reshape(img, (-1, 1, tool.height, tool.width))
        # 加载模型到指定设备
        model = CaptchaOCRModel()
        model.load_state_dict(torch.load(model_pth_path))
        model.to(self.__device)
        model.eval()
        # 进行预测
        outputs = model(reshape_img)
        lable = outputs.view(-1, len(tool.captcha_array))
        outputs_lable = CaptchaOCRDataset.vec2text(lable)
        logger.info(f'image_path:{image_path},predict:{outputs_lable}')
