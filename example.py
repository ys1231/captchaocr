from iyuecaptcha import tools
from iyuecaptcha.captchaocr import CaptchaOCR


def main():
    ocr = CaptchaOCR(5, tools.s_number + tools.s_uppercase, 150, 50)
    ocr.train('./data/tmp', './test.pth', train_path_split='_', epochs=100)
    ocr.predict('test/0G683_1712657652543.gif', './test.pth')
    # ocr.predicts('./data/test', './model1.pth', train_path_split='-')


if __name__ == '__main__':
    main()
