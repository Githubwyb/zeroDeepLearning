class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数以及激活函数
        激活函数的类型为 double -> double
        '''
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1, x2, x3, ...]和weights[w1, w2, w3, ...]打包在一起
        # 变成[(x1, w1), (x2, w2), (x3, w3), ...]
        # 然后利用map函数计算[x1 * w1, x2 * w2, x3 * w3, ...]
        # 最后利用reduce求和
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x * w,
                       zip(input_vec, self.weights)
                   ),
                   0.0
            ) + self.bias
        )
