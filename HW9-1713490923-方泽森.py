import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

P = D = 6 #初始化输入输出神经元的数量
max_iteration = 50 #最大迭代次数

#读入构成记忆规则的数据
memo_raw = pd.read_excel('Homework9.xlsx','In6_Memorizing')
memo = memo_raw.iloc[:,1:D+1].to_numpy() #读取输入的点
print('Memorizing data:\n',memo) #输出记忆的数据


weight = np.zeros((D,D)) #初始化权重矩阵为全零矩阵
for j in range(D):
    for p in range(D):
        if j == p:
            weight[j][p] = 0  #对角线上的权重为零
        else:
            weight[j][p] = np.dot(memo[:,j],memo[:,p]) #对应的两列数据做点积得到权重

print('\nweight:\n',weight) #输出权重矩阵

#联想数据
asso_data = pd.read_excel('Homework9.xlsx','In6_Associating')
x_total = asso_data.iloc[:,1:D+1].to_numpy() #读入需要联想的数据
print('\nAssociating Data:\n',x_total) #输出到控制台
print('\n')

data_num = 0 #表示当前推测的是第几条数据
#联想开始
for x in x_total: #对每一条数据进行联想
    net = np.dot(weight, x) #计算net值
    y = np.zeros((D,)) #设置y矩阵
    for _ in range(max_iteration): #根据net值对应转换y矩阵的值
        for i in range(D):
            if net[i] > 0:
                y[i] = 1
            elif net[i] < 0:
                y[i] = -1
            else:
                y[i] = x[i]

        if (x==y).all(): #判断y与x是否相等，如果是则输出，跳出对该条数据的联想，进行下一条
            print('Associating Data {0}:\n'.format(data_num + 1),x_total[data_num])
            print('Y:\n',y)
            print('*'*50,'\n')
            data_num += 1
            break
        else:
            x = y #如果y与x不想等，则将x更新为y,继续下一次迭代




