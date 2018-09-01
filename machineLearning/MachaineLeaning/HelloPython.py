import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import norm,poisson
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import CubicSpline
import math

if __name__ == "__main__":
    # a = np.arange(0,60,10).reshape((-1,1)) + np.arange(6)
    # print(a)

    # 1.使用array创建
    # 通过array函数传递List对象
    # L = [1, 2, 3, 4, 5, 6]
    # print(L)
    # a = np.array(L)
    # print(a)
    # print(a.shape)
    # # 创建多层嵌套的多为数组
    # b = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])
    # print(b)
    # b.shape = 4,3
    # print(b)
    # print(b.shape)
    # c = b.reshape((4,-1))
    # print(c)
    # a = np.array([1, 2, 3, 4])
    # print(a)
    #arrange支持大数据类型,arange函数类似于python的range函数，制定起始值，终止值和步长来创建数组，和range类似，arange同样不包括终值，但arange可以生成浮点类型
    # a = np.arange(1,200,0.5)
    # print(a)
    # linspace函数通过指定起始值，终止值和元素个数来创建数组，缺省值包括终止值
    # b = np.linspace(1,10,100,endpoint = False);
    # print( b)
    # 和linspace类似，logspace可以创建等比数列，下面函数代表起始值为10^1,有10个等比数列，终止值为10^2
    # d = np.logspace(1,2,10,endpoint=True)
    # print(d)
    # 下面创建起始值为2^0,终止值为2^10(包括),有11个数的等比数列
    # d = np.logspace(0, 10, 11, endpoint = True, base=2)
    # print(d)
    #在python里面没有'和”的区别
    # s = 'abcd'
    # g = np.fromstring(s, dtype=np.int8)
    # print(g)
    #存取
    #常规办法：数组元素的存取方法和python的标准方法相同
    # a = np.arange(10)
    #print(a)
    #切片[3,6)
    #print(a[3:6])
    #取某个函数
    #print(a[3])
    #省略下标，从0开始
    #print(a[:5])
    #可省略结束下表
    #print(a[3:])
    #步长为2
    #print(a[1:9:2])
    #步长为-1即翻转
    #print(a[::-1])
    #####切片数据是原数据的一个视图，与原数据共享内存空间，可以直接修改原数据值
    # a[1:4] = 10,20,30
    # print(a)
    # b = a[2:5]
    # b[0] = 100
    # print(a)
    ###########ab为不同对象的情况
    # a = np.logspace(0, 9, 10, base=2)
    # print(a)
    # i = np.arange(0,10,2)
    # print(i)
    # b = a[i]
    # print(b)
    # b[2] = 1.6
    # print(a)
    # print(b)

    # a = np.random.rand(10)
    # print(a)
    # print(a > 0.5)
    # b = a[a > 0.5]
    # print(b)
    # b[1] = 1
    # print(b)
    # print(a)


    # a = np.arange(0,60,10)  #行向量
    # b = a.reshape((-1,1))  #转换成列向量
    # print(b)
    # c = np.arange(6)
    # print(c)
    # f = b + c
    # print(f)
    #合并上述代码
    # a = np.arange(0,60,10).reshape((-1,1)) + np.arange(6)
    # print(a)
    # #二维数组的切片
    # print(a[[0,1,2,3],[2,3,4,5]])
    # print(a[3:,[0,2,5]])
    # i = np.array([True,False,True,False,False,True])
    # print(a[i])
    # print(a[i,3])


    #######python数学库与numpy的时间比较
    # j = 100000000
    # x = np.linspace(0,10,j)
    # start = time.clock()
    # y = np.sin(x)
    # t1 = time.clock() - start
    # print(t1)
    #
    # x = x.tolist()
    # start = time.clock()
    # for i, t in enumerate(x):
    #     x[i] = math.sin(t)
    # t2 = time.clock() - start
    # print(t2/t1)



    #绘图高斯分布
    # mpl.rcParams['font.sans-serif'] = [u'SimHei'] # 仿宋黑体
    # mpl.rcParams['axes.unicode_minus'] = False
    # mu = 0
    # sigma = 1
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 50)
    # y = np.exp(-(x - mu)**2 / (2 * sigma ** 2))/(math.sqrt(2*math.pi)*sigma)
    # plt.plot(x,y,'r-',x,y,'go',linewidth=2,markersize=8)
    # plt.grid(True)
    # plt.title(u"高斯分布")
    # plt.show()

    #损失函数
    x = np.array(np.linspace(start=-2, stop=3, num=1001, dtype=np.float))
    y_logit = np.log(1 + np.exp(-x))/math.log(2)
    y_boost = np.exp(-x)
    y_01 = x < 0
    y_hinge = 1.0 - x
    y_hinge[y_hinge < 0] = 0
    plt.plot(x, y_logit, 'r-', label='logistic loss', linewidth=2)
    plt.plot(x, y_01, 'g-', label='0/1 loss', linewidth=2)
    plt.plot(x, y_hinge, 'b-', label="hinge loss", linewidth=2)
    plt.plot(x, y_boost, 'm--', label='adaboost loss', linewidth=2)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
    # for i in range(10):
    #     print(i)
