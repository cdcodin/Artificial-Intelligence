import numpy as np
import matplotlib.pyplot as plt

#计算适合值并返回群体最佳解，个人最佳解
def calculate(x_total,P,G):
    #通过循环计算最后一列的适合值
    for i in range(m):
        x_total[i][D] = E(x_total[i,0:D])
        #更新个人最优解
        if x_total[i][D] < P[i][D]:
            P[i] = x_total[i]
    #寻找并其中的群体最佳解
    index = np.where(x_total == min(x_total[:,D])) #获得最小适合值在矩阵中的索引
    g = x_total[index[0]][0] #取最小适合值所在的行作为群体最佳解,x_total[index[0]] 后面再加一个[0]是为了让g变成一维数组
    #如果当前的群体最佳解适合值小于上一次的，则将其替换
    if g[D] < G[D]:
        G = g
    return x_total,P,G #将整个矩阵，个人最佳解，群体最佳解返回

#程序入口
while(True):
    index = int(input('请老师输入要查看的作业序号(1 or 2)，输入"0"退出：')) #获取老师的选择
    if index not in [1,2,0]:
        print('输入错误，请重新输入！\n')
        continue
    elif index == 0:
        break
    
    #1. 目标函数
    #2. 决策变数的上下界
    elif index == 1:
        X_UPPER = np.array([100,100]) #决策变量上界
        X_LOWER = np.array([0,0]) #决策变量下界
        target = [30,50]
        V_max = 5 #最大移动速度
        #目标函数 能量函数E
        def E(params):
            return (params[0] - 30)**2 + (params[1] - 50)**2

    elif index == 2:
        X_UPPER = np.array([6.28,6.28]) #决策变量上界
        X_LOWER = np.array([0,0]) #决策变量下界
        target = [4.7,4.7]
        V_max = 5 #最大移动速度
        #目标函数 能量函数E
        def E(params):
            return np.sin(params[0]) + np.cos(2*params[0]) + np.sin(params[1]) + np.cos(2*params[1])

    #3. 决定初始参数
    D = 2 #维度
    m = 30 #粒子总数
    W_init = 0.9 #初始权重
    W_final = 0.4 #终止权重
    c1 = 1 #认知参数
    c2 = 1.4 #社群参数
    V_init = 1 #初始参数
    V_max = 5 #最大移动速度
    # V_min = 0 #最小移动速度不设置
    Iter_final = 100 #范围为 [100,500]
    W_k = W_init #权重
    V_k = np.ones([m,D]) * V_init #产生速度

    #4. 随机产生目前解,并计算适合值（放到最后一列上）
    x_total = np.random.rand(m,D+1)  #目前解
    # P = x_total[:,:D] #取目前解的前D列作为个人最优解
    P = x_total #初始的个人最佳解就是目前解
    for i in range(m):
        P[i,D] = float("inf") #将最后一列的适合值赋值为无穷大，方便后续调用函数calculate进行比较
    G = x_total[0] #初始化群体最佳解
    G[D] = float("inf")
    x_total,P,G = calculate(x_total,P,G) #第一次初始化操作

    #5. 开始迭代
    for k in range(Iter_final):
        #速度计算
        for i in range(m):
            for j in  range(D):
                x1 = c1*np.random.random()*(P[i][j] - x_total[i][j])
                x2 = c2*np.random.random()*(G[j] - x_total[i][j])
                # print(V_k[i][j])
                V_k[i][j] = W_k * V_k[i][j] + x1 + x2
                #判断速度是否超过最大速度
                if V_k[i][j] > V_max:
                    V_k[i][j] = V_max
                elif V_k[i][j] < -V_max:
                    V_k[i][j] = -V_max

                #移步
                x_total[i][j] = x_total[i][j] + V_k[i][j]
                #判断是否超出边界
                if x_total[i][j] > X_UPPER[j]:
                    x_total[i][j] = X_UPPER[j]
                    V_k[i][j] = V_init
                elif x_total[i][j] < X_LOWER[j]:
                    x_total[i][j] = X_LOWER[j]
                    V_k[i][j] = V_init

        #调用函数计算
        x_total,P,G = calculate(x_total,P,G)

        #修正权重W
        W_k = W_init - (k/Iter_final)*(W_init - W_final)

        #画图
        plt.ion()
        plt.cla()
        plt.xlim(X_LOWER[0],X_UPPER[0]) #确定坐标的上下界
        plt.ylim(X_LOWER[1],X_UPPER[1])
        plt.scatter(target[0],target[1],color='r') #画出目标点
        x = x_total[:,0] #取x_total中的第一维度
        y = x_total[:,1] #取x_total中的第二维度        
        plt.title(f'Iteration {k+1}')
        plt.scatter(x,y,color='b')
        plt.pause(0.01)        
        plt.show()
        plt.ioff()


    print('群体最佳解为G:\n',G)
