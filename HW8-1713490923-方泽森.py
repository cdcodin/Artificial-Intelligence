import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#计算两点间的距离
def dis(x,y):
    res = 0
    for i in range(D):
        res += (x[i]-y[i])**2
    return np.sqrt(res)

#参数初始化
D = 2 #输入神经元数
P = 20 #输出神经元数
R = 20 
ita = 0.05 #学习率
alpha = 0.9 
lamda = 0.9 #学习率衰减率
ita_min = 0.001 #最小学习率
iter = 0 #迭代次数
iter_max = 20 #最大迭代次数

raw_data = pd.read_excel('Homework8.xlsx','C4') #读取文件
x_total = raw_data.iloc[:,1:3].to_numpy() #读取输入的点
weight = np.random.random((P,P,2)) #随机生成输出神经元的初始权重

#若要使用有序的点进行训练，请将下面这段代码的注释取消
# for i in range(P):
#     for j in range(P):
#         weight[i][j] = [0.2+(i*0.03),0.2+(j*0.03)]

#训练开始
for i in range(iter_max):   
    #对每一个样本点都进行操作
    for dot in x_total: 
        #计算winner center
        dis_min = 10 #假定初始的最小距离，10 远远大于本题中点间的距离
        p_win = 0 #记录winner center的p
        q_win = 0 #记录winner center的q
        for p in range(P):
            for q in range(P):
                dist = dis(weight[p][q],dot)
                if dist < dis_min:
                    dis_min = dist
                    p_win = p
                    q_win = q   
        #修改权重
        for p in range(P):
            for q in range(P):
                dis_cc2wc = dis(weight[p][q],weight[p_win][q_win])
                weight[p][q] = weight[p][q] + ita * (dot - weight[p][q]) *np.exp(-(dis_cc2wc/R)**2)

    
    #对所有点迭代完一次后（即一个周期结束），修改一次学习率等参数
    R = R * alpha
    ita = ita * lamda
    if ita < ita_min:
        ita = ita_min 

    #画图
    plt.ion()
    plt.cla()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title(f'Iteration {i+1}')
    plt.scatter(raw_data.iloc[:,1],raw_data.iloc[:,2],color='b') #原始数据点使用蓝色标出
    plt.scatter(weight[:,:,0],weight[:,:,1],color='r') #训练后的权重使用红色标出
    plt.pause(0.001)
    plt.show()
    plt.ioff()
#训练结束
input('\n输入任意值查看分布结果\n')
#对训练结果进行归类，记录每个中心包含的点数
cluster = np.zeros((P,P)) #初始化center所包含的点的数量的矩阵，所有center初始包含的点都为零
for dot in x_total:
    dis_min = 10
    center_p = 0
    center_q = 0
    #找到距离最近的center
    for p in range(P):
        for q in range(P):
            dist = dis(weight[p][q],dot)
            if dist < dis_min:
                center_p = p
                center_q = q
                dis_min = dist
    cluster[center_p][center_q] += 1 #找到该点所对应的最近的center，是它包含的点的数量加1
print(cluster) #将结果集输出

#画出聚类结果图
distubution = []
for p in range(P):
    for q in range(P):
        if cluster[p][q] != 0:
            distubution.append(weight[p][q]) #将非死亡节点记录到分布的列表中
#画图
plt.ion()
plt.cla()
plt.scatter([dot[0] for dot in distubution],[dot[1] for dot in distubution]) #使用分布的列表画图对应的点
input('\n输入任意键结束程序\n')
plt.show()
plt.ioff()