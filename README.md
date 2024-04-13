# 验证码识别

- 基于`PyTorch`训练模型

# install

```shell
# 安装
pip install torch torchvision # torchaudio
# 数据集生成
pip install Pillow captcha
# 模型部署
#pip install flask
pip install tensorboard
```
# 使用

```python
# 没有数据先生成数据 
## 生成训练的验证码 tools.py
generate_captcha(path='../data/train', captcha_lenth=5, lenth=10000, font_sizes=(24,),width=150, height=50)
# 生成测试的验证码
generate_captcha(path='../data/test', captcha_lenth=5, lenth=100, font_sizes=(24,),width=150, height=50)

# example.py 
ocr = CaptchaOCR(5, tools.s_number + tools.s_uppercase, 150, 50)
## 训练预测
ocr.train('./data/tmp', './model1.pth', train_path_split='_',epochs=100)
## 单个预测
ocr.predict('data/tmp/0G683_1712657652543.gif', './model1.pth')
## 多个预测
ocr.predicts('./data/test','./model1.pth', train_path_split='-')


```

# 补充
- 如果数据少可增加训练次数
- 训练如果只有CPU很慢,建议使用GPU已经自适应前提是GPU能用
```shell
# Mac 字体路径 生成验证码图片需要
/System/Library/Fonts/Supplemental/Arial.ttf

```

# 参考
