from perceptron import Perceptron
from functools import reduce  # py3


def f(x):
    '''
    定义激活函数
    '''
    return 1 if x > 0 else 0


def get_train_dataset():
    '''
    基于and真值表构建训练数据
    '''
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1, 1] -> 1, [0, 0] -> 0, [1, 0] -> 0, [0, 1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():
    '''
    使用and真值表训练感知器
    '''
    # 创建感知器，输入参数个数为2(因为and是二元函数),激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10轮，学习速率为0.1
    input_vecs, labels = get_train_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return p


if __name__ == '__main__':
    # 训练and感知器
    and_perceptron = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perceptron)
    # 测试
    print('1 & 1 = %d' % and_perceptron.predict([1, 1]))
    print('0 & 0 = %d' % and_perceptron.predict([0, 0]))
    print('1 & 0 = %d' % and_perceptron.predict([1, 0]))
    print('0 & 1 = %d' % and_perceptron.predict([0, 1]))
