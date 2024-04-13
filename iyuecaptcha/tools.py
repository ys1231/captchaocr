import time
import os.path
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class Tools:
    # 验证码长度5
    captcha_lenth: int = 5
    # 验证码个数
    lenth: int = 100
    # 验证码范围
    captcha_array: list = field(
        default_factory=lambda: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                                 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                                 'Y', 'Z'])
    # 验证码 height
    width: int = 150
    # 验证码 height
    height: int = 50


s_number = "0123456789"
s_lower = "abcdefghijklmnopqrstuvwxyz"
s_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
s_symbol = "!@#$%^&*()_+-=<>?/.,;:|"


def generate_captcha(path: str, captcha_lenth: int, lenth,
                     fonts: list = ['/System/Library/Fonts/Supplemental/Arial.ttf'],
                     font_sizes: tuple[int, ...] = (12,), width=110, height=35):
    """
    生成验证码
    :param path: 保存路径
    :param captcha_lenth: 验证码长度
    :param lenth: 验证码个数
    :param fonts: 使用字体列表 [ttf,]
    :param font_sizes: 设置生成的字体大小 (12,)
    :param width: 图片验证码的宽度
    :param height: 图片验证码高度
    """
    import random
    from captcha.image import ImageCaptcha

    use_s = s_number + s_uppercase
    logger.debug(f'use_s is {use_s}')

    os.makedirs(path, exist_ok=True)
    start = time.time()
    for i in range(lenth):
        # 获取一组不重复的随机样本
        image_val = "".join(random.sample(s_number + s_uppercase, k=captcha_lenth))
        # 设置使用的字体 验证码宽高
        image = ImageCaptcha(fonts=fonts, font_sizes=font_sizes, width=width, height=height)
        image.write(image_val, os.path.join(path, f'{image_val}-{int(time.time())}.png'))
        if i % 100 == 0 and i > 0:
            logger.info(f'generate {i} captcha success')
        # logger.debug(f'generate {os.path.join(path, f"{image_val}-{int(time.time())}.png")} success')
    end = time.time()

    logger.info(f'generate {lenth} captcha cost {end - start} seconds')


def get_defualt_device():
    """
    获取默认设备，优先选择 CUDA（如果可用），其次是 MPS（如果可用），最后是 CPU。
    返回: device (torch.device): 返回一个torch.device对象，代表当前可用的最好设备。
    """

    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # 检查CUDA是否可用，如果可用，则使用CUDA设备
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
        # 检查MPS（Mac Pro GPU）是否可用，如果可用，则使用MPS设备
    else:
        device = torch.device('cpu')
        # 如果CUDA和MPS均不可用，则使用CPU设备
    logger.debug(f'current device is {device}')
    return device


def main():
    # 生成训练的验证码
    generate_captcha(path='../data/train', captcha_lenth=5, lenth=10000, font_sizes=(24,), width=150, height=50)
    # 生成测试的验证码
    generate_captcha(path='../data/test', captcha_lenth=5, lenth=100, font_sizes=(24,), width=150, height=50)
    pass


if __name__ == '__main__':
    main()
