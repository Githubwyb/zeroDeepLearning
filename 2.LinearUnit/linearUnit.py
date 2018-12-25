from perceptron import Perceptron

# 定义激活函数F

f = lambda x: x

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''
        初始化线性单元，设置输入参数的个数
        '''
        Perceptron.__init__(self, input_num, f)
