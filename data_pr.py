# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 15:24
# @Author  : dave
# @Email   : yushan2025@gmail.com
# @File    : data_pr.py
# @Software: PyCharm
import numpy as np
import random,csv
import time
def changeSentence(input):
    if input[len(input) - 2] == '1':
        output0 = "YES"
    else:
        output0 = "NO"
    output1 = input[0:len(input) - 4]
    return output0 + '/' + '"' + output1 + '"'

def preHandle():
    source_file = 'review_0408.txt'
    handled_file ='review_0408_handled0.txt'
    data_to_flie=open(handled_file, 'w',newline='')
    i=1
    with (open(source_file,'r', encoding='UTF-8')) as f:
        for line in f.readlines():
            if line!='\n':
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                print("\n",i,".From\n")
                i=i+1
                print(line)
                print("\nto\n")
                print(changeSentence(line))
                print(changeSentence(line), file=data_to_flie)
        data_to_flie.close()

if __name__ == '__main__':
    start_time=time.clock()
    preHandle()
    # print staut_list
    end_time=time.clock()
    print(time.perf_counter)
    print("Running time:",(end_time-start_time))  #输出程序运行时间
