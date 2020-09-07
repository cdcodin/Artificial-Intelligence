import numpy as np
import xlrd
import xlwt
from matplotlib import pyplot as plt

class adaline:
    """
    adaline : 输入参数初始化后，使用train方法训练模型，使用draw方法画图
    """
    def __init__(self,x_total,t,w1=[10.0,-1.0],ita=0.01,):
        """
        Args : 
            x_total : 包含x0,x1
            t : target值
            w1 : 初始权重的默认值为建模问题的初始权重
            ita ：学习效率默认值同理        
        """
        self.x_total = x_total
        self.t = t
        self.ita = ita #初始化学习效率
        self.w1 = np.array(w1).T #初始权重
        self.wlst = []  #记录每一次迭代后的权重，作为画图的依据
        self.iter = 0   
        self.Max_iteration = 50
        self.k = 0

        self.W_best = self.w1 #使用二分类的时候才会用到这两个属性
        self.CCount_max = 0
    
    def train(self):
        wk = self.w1 #第一次权重为初始权重
        while self.iter < self.Max_iteration:
            self.iter += 1   #迭代开始

            for xk,tk in zip(self.x_total,self.t):
                y = np.dot(xk,wk) #计算y值
                wk = wk + self.ita * (tk - y)*xk #修改权重
                self.k += 1
             
            self.wlst.append(wk) #一次iteration结束，记录一次权重  

    def classify(self):
        wk = self.w1
        while self.iter < self.Max_iteration:
            self.iter += 1   

            for xk,tk in zip(self.x_total,self.t):
                y = np.dot(xk,wk)
                wk = wk + self.ita*(tk - y)*xk
                
                #计算正确率
                CCount = 0
                for x,t_k in zip(self.x_total,self.t):
                    y = np.dot(wk,x)
                    if t_k == 1 and y > 0:
                        CCount = CCount + 1
                    elif t_k == -1 and y < 0:
                        CCount = CCount + 1

                # 更新最佳权重和最大正确数
                if CCount > self.CCount_max:
                    self.W_best = wk
                    self.CCount_max = CCount
                    self.wlst.append(wk)

                self.k += 1

    def draw(self,xlim=(-2,10),ylim=(-2,10),clas=False): #clas标识画图是否为二分类图
        fig,ax = plt.subplots()
        if self.x_total.shape[1] == 2:
            ax.scatter(self.x_total[:,1],self.t,color='blue') #将数据点描上
        elif self.x_total.shape[1] == 3:
            ax.scatter(self.x_total[:,1],self.x_total[:,2],color='blue')
        plt.title('Adaline') #设置标题
        plt.xlim(xlim[0],xlim[1]) #设置x,y轴坐标区域
        plt.ylim(ylim[0],ylim[1])
        x1 = np.linspace(-5,10,30) #取模型在x上的取值范围
        
        
        if clas: #当clas为True时，使用以下的权重绘制分类图
            for w in self.wlst:
                x2 = (-w[1]/w[2])*x1 - w[0]/w[2] #由权重得到x2的值s
                ax.plot(x1,x2,color='orange')
                plt.pause(0.1)           
                ax.lines.pop(0) #循环未结束前，将这次画的直接删除
            x2 = (-w[1]/w[2])*x1 - w[0]/w[2] #画出最佳的权重
            ax.plot(x1,x2,color='orange')
        else: #绘制建模图形
            for w in self.wlst:
                x2 = w[0]+w[1]*x1 #由权重得到x2的值
                ax.plot(x1,x2,color='orange')
                plt.pause(0.1)           
                ax.lines.pop(0) #循环未结束前，将这次画的直接删除
            x2 = self.wlst[-1][0]+self.wlst[-1][1]*x1 #画出最佳的权重
            ax.plot(x1,x2,color='orange')

        plt.show() #显示图形

# 从指定文件中读取数据并转换成矩阵,并初始化
def get_data(filename,sheet_name,x=1):
    """获取数据.
    Args : 
          filename : execel文件名
          sheet_name : 表名
          x : 需要读取的x_total的列数
    return : x_total,t   
    """
    #读取数据
    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_name(sheet_name)

    # 获取表中的有效行数和列数，作为构造矩阵的参数
    row = sheet.nrows -1
    col = sheet.ncols
    x0 = np.ones((row,1))   #构建x0

    # 利用循环将表中的每个单元格赋值到矩阵中
    temp_matrix = np.zeros([row,col])
    for i in range(row):
        for j in range(col):
            temp_matrix[i][j] = sheet.cell_value(i+1,j) #从i + 1开始，跳过表头
    if x == 1: #当x == 1时，读取的是表2和表3
        matrix = np.c_[x0,temp_matrix]  #将 x0 与 x1,t 连接起来
        x_total = matrix[:,[0,2]] #取x0,x1
        t = matrix[:,3] #取t
    elif x == 2: #当x == 2时，读取的是表1
        matrix = np.c_[x0,temp_matrix]  #将 x0 与 x1,t 连接起来
        x_total = matrix[:,[0,2,3]] #取x0,x1,x2
        t = matrix[:,4] #取t

    return x_total,t

def save(filename,sheetname,result):
    workbook = xlwt.Workbook() #新建工作薄
    work_sheet = workbook.add_sheet(sheetname) #添加新表
    work_sheet.write(0,0,'W_best') #设置表头
    work_sheet.write(0,1,'bias_best')
    w = ''
    for i in range(0,result.shape[0]):
        w += f'{result[i]} '
    work_sheet.write(1,0,w) #输入要保存的数据
    work_sheet.write(1,1,result[0])    
    workbook.save(filename) #保存工作薄