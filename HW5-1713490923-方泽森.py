import numpy as np
import xlrd
import xlwt
from matplotlib import pyplot as plt

import HW5_1713490923_方泽森_adaline as ad #导入adaline模块


while True:
    option = int(input('请老师输入您想要查看的作业序号（与表顺序相同）,输入0退出程序：'))
    if option == 1:
        x_total,t = ad.get_data('Homework5.xlsx','HW_5_CP',x=2) #使用adaline模块中的读取数据方法
        adaline = ad.adaline(x_total,t,ita=0.1,w1=[-1.2,-0.1,1.009]) #输入数据，创建一个pocke对象
        adaline.classify()
        print('W_best:',adaline.W_best)
        adaline.draw((-3,2),(-3,2),clas=True)
        ad.save('HW_5_CP_W-best.xls','HW_5_MP_Linear',adaline.wlst[-1]) #使用adaline模块中的存储数据方法

    elif option == 2:
        x_total,t = ad.get_data('Homework5.xlsx','HW_5_MP_Linear') #从execel表格中读取输入并完成初始化
        adaline = ad.adaline(x_total,t) #输入数据，创建一个adaline对象
        adaline.train() #调用对象的train方法训练模型，找出最佳权重
        print('W_best:',adaline.wlst[-1]) #在控制台中输出权重
        adaline.draw()    #调用draw方法画图
        ad.save('MP_Linear_W_best.xls','HW_5_MP_Linear',adaline.wlst[-1]) #调用save方法保存最佳权重

    elif option == 3:
        x_total,t = ad.get_data('Homework5.xlsx','HW_5_MP_un-Linear')
        adaline = ad.adaline(x_total,t) #在创建对象时输入画图时x,y轴的限制
        adaline.train()
        print('W_best:',adaline.wlst[-1])
        adaline.draw(xlim=(0,1),ylim=(-0.5,3))
        ad.save('MP_un-Linear_W_best.xls','HW_5_MP_Linear',adaline.wlst[-1]) #调用save方法保存最佳权重

    elif option == 0:
        break
