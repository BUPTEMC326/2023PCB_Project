#encoding:utf8
import os
import re
import json
import codecs
import threading
from py2neo import Graph, NodeMatcher
import pandas as pd
import numpy as np
from tqdm import tqdm


from py2neo import Graph, Node, Relationship
import pandas as pd
import csv
# 连接neo4j数据库，输入地址、用户名、密码
# graph = Graph("http://localhost:7474", username="neo4j_test1", password='123456')
graph = Graph("http://localhost:11004", username="PCB", password='123456')
graph.delete_all()

with open(r'C:\Users\LWQ\Desktop\liwanqing\ER\entity extraction\KBQA-for-Diagnosis-main\knowledge_extraction\Graph_Data_ALL.csv','r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)
print(data[1])

node = Node('head', name=data[1][1])
node2 = Node('tail',name= data[1][3])

graph.create(node)
graph.create(node2)

with open(r'C:\Users\LWQ\Desktop\liwanqing\ER\entity extraction\KBQA-for-Diagnosis-main\knowledge_extraction\Graph_Data_ALL.csv','r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)

#遍历节点
for i  in  range(1,len(data)):
    matcher = NodeMatcher(graph)
    #nodelist[0]就代表当前遍历节点
    nodelist = list(matcher.match('head', name=data[i][1]))
    if len(nodelist) > 0:
        matcher = NodeMatcher(graph)
        nodelist1 = list(matcher.match('tail', name=data[i][3]))
        if len(nodelist1)>0:
            #尤为重要，采用nodelist[0]会跳过创建节点而直接建立联系，
            # 如果先node = ('人物1', name=data[i][0]),再Relationship(node, data[i][2], node2)还是会重复建立节点
            zhucong = Relationship(nodelist[0], data[i][2], nodelist1[0])
            graph.create(zhucong)

        else:
            cong = Node('tail', name=data[i][3])
            graph.create(cong)
            zhucong = Relationship(nodelist[0], data[i][2], cong)
            graph.create(zhucong)
    else:
        zhu = Node('head', name=data[i][1])
        graph.create(zhu)
        matcher = NodeMatcher(graph)
        nodelist1 = list(matcher.match('tail', name=data[i][3]))
        if len(nodelist1)>0:
            # print('1没有2有')
            zhucong = Relationship(zhu, data[i][2], nodelist1[0])
            graph.create(zhucong)
        else:
            cong = Node('tail', name=data[i][3])
            graph.create(cong)
            zhucong = Relationship(zhu, data[i][2], cong)
            graph.create(zhucong)


if __name__ == '__main__':
     path = r"C:\Users\LWQ\Desktop\liwanqing\ER\entity extraction\KBQA-for-Diagnosis-main\knowledge_extraction/Graph_Data_ALL.csv"
     # call apoc.export.csv.query("MATCH (n)-[r]-(m) return n,r,m","d:/movie.csv",{format:'plain',cypherFormat:'updateStructure'})
