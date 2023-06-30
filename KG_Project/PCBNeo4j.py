#encoding:utf8
import json
from py2neo import Graph, NodeMatcher
from py2neo import Graph, Node, Relationship
from HDMI_function import *
from Camera import *
from Button import *
from IC import *
from CLK  import *
from MicrostripLine import *
from SwitchPower import *

import config as c
import params as p
import os




# 1 ---------------------------------------------------------①功能函数部分--------------------------------------------------------
# neo4j查找节点函数
def neo4j_searcher(cql_list):
    ress = ""
    if isinstance(cql_list, list):
        for cql in cql_list:
            rst = []
            data = graph.run(cql).data()
            if not data:
                continue
            for d in data:
                d = list(d.values())
                if isinstance(d[0], list):
                    rst.extend(d[0])
                else:
                    rst.extend(d)

            data = "、".join([str(i) for i in rst])
            ress += data + "\n"
    else:
        data = graph.run(cql_list).data()
        if not data:
            return ress
        rst = []
        for d in data:
            d = list(d.values())
            if isinstance(d[0], list):
                rst.extend(d[0])
            else:
                rst.extend(d)

        data = "、".join([str(i) for i in rst])
        ress += data

    return ress

# 建关系的时候需要 获取节点
#eg: type="IO",name="IO"
def getNode(type,name):
    # IO节点
    matcher = NodeMatcher(graph)
    nodelist = list(matcher.match(type, name=name))
    # print(nodelist)
    lenIO=len(nodelist)
    if lenIO>0:
        return nodelist[0]
    else:
        node = Node(type, name=name)
        graph.create(node)
        return node

# 从cql获取函数function
def chpheToFunction(cql):
    # eg：cql="match(n) where n.name='IO' set n.label='IOfun1' return n"
    # 查找IO节点
    cql_template = cql
    answer = neo4j_searcher(cql_template)
    # (_26:IO {height: 198, label: '\u79d1\u6bd4', name: 'IO', position: '\u5f97\u5206\u540e\u536b', prop1: 'f1()', prop2: 'f2()'})
    # print(answer)
    left = answer.split("{")
    right = left[1].split("}")
    arr = []
    right = right[0]
    left = right.split('\'')
    # print(left)

    for i in range(1, len(left), 2):
        # print(left[i-1])
        # print(type(left[i]))
        if str(left[i-1]) in ', name:  ':
            continue
        arr.append(left[i])
    return arr
# ---------------------------------------------------------①end---------------------------------------------------------






# 2------------------------------------②连接neo4j数据库，输入地址、用户名、密码---------------------------------
graph = Graph("http://localhost:11004", username="PCBNeo4j", password='123456')
graph.delete_all()
# ------------------------------------------------------------②end--------------------------------------







# 3--------------------------------------------③建立节点部分------------------------------------------

# def NoUse:
    # 创建IOnode(leiixng,shuxing)
# for key in c.cql["IO"]:
#     nodeHighSpeed=Node('IO', name=key)
#     for k in c.cql["IO"][key]:
#         if(k=='cql_template'):
#             continue
#         nodeHighSpeed[k]=c.cql["IO"][key][k]
#     graph.create(nodeHighSpeed)
#
# nodeIC=Node('IC',name="IC",prop1="NCPIN")
# graph.create(nodeIC)
#
#
# for key in c.cql["HighSpeedComp"]:
#     nodeHighSpeed=Node('HighSpeedComp', name=key)
#     for k in c.cql["HighSpeedComp"][key]:
#         if(k=='cql_template'):
#             continue
#         nodeHighSpeed[k]=c.cql["HighSpeedComp"][key][k]
#     graph.create(nodeHighSpeed)
#
# for key in c.cql["TranLine"]:
#     nodeHighSpeed=Node('TranLine', name=key)
#     for k in c.cql["TranLine"][key]:
#         if(k=='cql_template'):
#             continue
#         nodeHighSpeed[k]=c.cql["TranLine"][key][k]
#     graph.create(nodeHighSpeed)

for i in c.cql:
    for key in c.cql[i]:
        nodeHighSpeed = Node(i, name=key)
        for k in c.cql[i][key]:
            if (k == 'cql_template'):
                continue
            nodeHighSpeed[k] = c.cql[i][key][k]
        graph.create(nodeHighSpeed)


# --------------------------------------------------------------------------------------------------------




# 4------------------------------------------------④建关系---------------------------------------------------
# nodeGPU=getNode("GPU","GPU")
# nodeIO=getNode("IO","IO")
# print(nodeIO)
# rel1 = Relationship(nodeGPU, "f5()", nodeIO)
# graph.create(rel1)
# ------------------------------------------------------------------------------------------------------------






# 6 -----------------------------------------------⑥修改成板子提取出来的参数---------------



try:
    # CLK = CLK('1',1,1,1,1)
    CLK = CLK(1.0, 1.0, 1.0)
    HDMI = HDMI(p.ESDProDev, p.filterDev)
    Camera = Camera(p.ESDProDev, p.filterDev)
    Button = Button(p.ESDProDev, p.filterDev)
    IC = IC(p.NCpin)
    SwitchPower=SwitchPower(1.0, 1.0, 1.0, 1.0, 1.0)
    MicrostripLine=MicrostripLine(1.0, 1.0, 1.0, 1.0)
except Exception as e:
    print(e,'变量输入有误，请输入数值类型')
    os._exit(0)
#------------------------------------------------------------------------------------




#7 ------------------------------------⑦真正的功能判断模块（自动实现，无需改动）----------------------------------
for k in c.cql:
    for key in c.cql[k]:
        # -------------------------------------------------通过cql查询函数-----------------------------------
        funHDMI=chpheToFunction(c.cql[k][key]['cql_template'])
        print(funHDMI)
        for f in funHDMI:
            function=getattr(eval(key),f,None)
            # print(function)
            print(key,':',function())
# ----------------------------------------------------------------------------------------



