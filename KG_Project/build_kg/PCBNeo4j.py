#encoding:utf8
import json
from py2neo import Graph, NodeMatcher
from py2neo import Graph, Node, Relationship




# ---------------------------------------------------------功能函数部分--------------------------------------------------------
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
    print(nodelist)
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
    print(left)

    for i in range(5, len(left), 2):
        print(left[i])
        print(type(left[i]))
        arr.append(left[i])
    return arr





# ------------------------------------连接neo4j数据库，输入地址、用户名、密码---------------------------------
graph = Graph("http://localhost:11004", username="PCBNeo4j", password='123456')
graph.delete_all()






# --------------------------------------------建立节点和关系部分------------------------------------------
# 创建IOnode(leiixng,shuxing)
nodeIO=Node('IO',name="HDMI",prop1="f_ESDAndFilter(ESDprot,filter)",prop2="f2()")
graph.create(nodeIO)





# ------------------------------------------------------------------建关系---------------------------------
nodeGPU=getNode("GPU","GPU")
nodeIO=getNode("IO","IO")
# print(nodeIO)
rel1 = Relationship(nodeGPU, "f5()", nodeIO)
graph.create(rel1)



# ------------------------------------------------------------通过cql查询函数-----------------------------------
cql = "match(n) where n.name='HDMI'  return n"
fun=chpheToFunction(cql)
print(fun)
# [f_ESDAndFilter(ESDprot,filter),f2()]


