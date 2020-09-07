import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def forecast(weight_in,weight_out):
    nets = [np.dot(weight,x) for weight in weight_in] #计算每一个隐藏层神经元对应的net值
    HL = [sigmoid(net) for net in nets] #计算对应的sigmoid值
    HL = np.array(HL) #将列表转换为可积的类型
    print('HL : ',HL)
    HL = np.insert(HL,0,1) #加入h0到隐藏层中

    y = np.dot(HL,weight_out) #计算预测值y

    print('\ninput data x1 : ',x1) #输出x1的值
    print('\noutput y : ',y) #输出y的值


def sigmoid(net):
    return 1 / ( 1 + np.exp(-net)) #计算sigmoid函数


x1 = input('x1 : ')  #通过控制台输入x1
x = np.array([1,float(x1)]) #加入输入神经元 x0 = 1

raw_1 = pd.read_excel('Homework6.xlsx','Weight_In_Hid') #读取文件
weight_in = [raw_1.iloc[:,i].to_numpy() for i in range(1,3)] #读取每个输入神经元对应的权重

raw_2 = pd.read_excel('Homework6.xlsx','Weight_Hid_Out') #读取文件
weight_out = raw_2.iloc[:,1].to_numpy() #读取每个隐藏神经元对应的权重

print('\nWeight_In_Hid : \n',raw_1.iloc[:,1:3].to_numpy())
print('\nWeight_Hid_Out : \n',raw_2.iloc[:,1].to_numpy())

forecast(weight_in,weight_out) #调用forcast预测y值

hid_neus = int(input('\n输入隐藏层神经元数：'))
x1 = input('x1 : ')
x = np.array([1,float(x1)])

#随机生成权重
weight_in = np.random.random((hid_neus,2)) # 2对应着两个输入神经元
weight_out = np.random.random((hid_neus + 1,)) #因为输出神经元只有一个，故只需给出隐藏神经元数量的权重即可

forecast(weight_in,weight_out)