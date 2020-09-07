import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def sigmoid(net):
    return 1 / ( 1 + np.exp(-net)) #计算sigmoid函数

#各层神经元数量
in_neus = 1
hid_neus = int(input('\n输入隐藏层神经元数：'))
out_neus = 1

#设置学习率和惯性率(试错求)
eta_HL = 0.2 #大于0.1，隐藏层学习率
eta_OL = 0.05 #小于0.1，输出层学习率
alpha = 0.1  #惯性率
iter = 0 #迭代周期
Max_iteration = 20 #最大迭代周期

raw_1 = pd.read_excel('Homework7.xlsx','In1Out1') #读取文件
x1 = raw_1.iloc[:,1].to_numpy() #读取x1
x0 = np.ones(len(x1)) #x0
x = np.vstack((x0,x1)) #插入x0
X_total = np.array(x).T 
T = raw_1.iloc[:,2].to_numpy() #读取T

#根据神经元数随机生成权重
w_in = np.random.random((in_neus + 1,hid_neus)) # 2 *2
w_out = np.random.random((hid_neus + 1,out_neus))  # 3 * 1
del_w_I2H = np.zeros((in_neus + 1,hid_neus))
del_w_H2O = np.zeros((hid_neus + 1,out_neus))


#开始迭代
m = len(X_total)
for iter in range(Max_iteration):
    
    for c in range(m):

        x_k = X_total[c]
        net_h =  np.dot(x_k,w_in)
        HL = [sigmoid(net) for net in net_h] #计算对应的sigmoid值
        HL = np.array(HL) #将列表转换为可积的类型
        HL = np.insert(HL,0,1) #加入h0到隐藏层中
        y = np.dot(HL,w_out) #这里的 net 值就是 y 值
        
        delta_o = [T[c] - y_ for y_ in y]
        delta_h = [] #通过循环计算每个h对应的delta值
        for i in range(1,hid_neus+1):
            h_ = 0
            for j in range(out_neus):
                h_ += w_out[i][j]*delta_o[j] 
            delta_h.append(h_ * HL[i] * ( 1 - HL[i] )) 

        #修改隐藏层到输出层要改变的权重
        for p in range(hid_neus+1):
            for q in range(out_neus):
                del_w_H2O[p][q] = alpha*del_w_H2O[p][q]  + eta_OL*delta_o[q]*HL[p]
        
        #修改输入层到隐藏层要改变的权重
        for j in range(in_neus+1):
            for p in range(hid_neus):
                del_w_I2H[j][p] = alpha*del_w_I2H[j][p] + eta_HL*delta_h[p]*X_total[c][j]
        
        #修改权重的值
        w_in = w_in + del_w_I2H
        w_out = w_out + del_w_H2O
    

    #画底图
    raw_2 = pd.read_excel('Homework7.xlsx','Drawing_In1Out1') #读取文件
    x2 = raw_2.iloc[:,1].to_numpy()
    T2 = raw_2.iloc[:,2]
    x = np.vstack((x0,x2))
    x2 = np.array(x).T

    #计算需要当前权重下的输出值,记录到y_set中，计算过程与上面迭代的过程相似
    y_set = []
    for x in x2:
        net_h =  np.dot(x,w_in)
        HL = [sigmoid(net) for net in net_h] 
        HL = np.array(HL) 
        HL = np.insert(HL,0,1) 
        y = np.dot(HL,w_out) 
        y_set.append(y[0])
    
    x2 = raw_2.iloc[:,1]
    plt.ion()
    plt.cla()
    plt.xlim(-10,10)
    plt.ylim(-1,4)
    plt.title(f'Iteration {iter+1}')
    plt.plot(x2,T2,color='r')
    plt.plot(x2,y_set,color='b')
    plt.pause(0.001)
    plt.show()
    plt.ioff()

#将权重输出到控制台中
print('\nWeight_In_Hidden:\n',w_in)
print('\nWeight_Hidden_Out:\n',w_out)

#将权重存储到excel中
writer = pd.ExcelWriter('./Weight.xlsx')
w_in = pd.DataFrame(w_in)
w_in.to_excel(writer,'Weight_Hidden_Out',index=False,header=False)
w_out = pd.DataFrame(w_out)
w_out.to_excel(writer,'Weight_In_Hidden',index=False,header=False)
writer.save()
writer.close()