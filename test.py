class test(object):
    def __init__(self):
        self.str = '123'

def change(obj):
    obj.str = '456'

if __name__ == '__main__':
    a = test()
    print(a.str)
    change(a)
    print(a.str)
