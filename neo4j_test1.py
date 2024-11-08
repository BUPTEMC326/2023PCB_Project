from py2neo import Graph, NodeMatcher, Node, Relationship
import csv
from py2neo import Graph, Node, Relationship, NodeMatcher
import csv

# 连接到Neo4j数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

try:
    with open(r'C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\data\Graph_Data_ALL（1）.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
except UnicodeDecodeError:
    with open(r'C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\data\Graph_Data_ALL（1）.csv', 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        data = list(reader)

# 打印每一行的长度以调试
for idx, row in enumerate(data):
    print(f"Row {idx} length: {len(row)} -> {row}")

# 创建节点和关系
matcher = NodeMatcher(graph)

for i in range(1, len(data)-4):
    row = data[i]
    nodes = []
    relations = []

    # 处理节点和关系（假设每行格式是 node1, relation1, node2, relation2, node3, relation3, node4, ...）
    for j in range(1, len(row)):
        if row[j]:
            if j % 2 == 1:  # 奇数索引是节点
                nodes.append(row[j])
            else:           # 偶数索引是关系
                relations.append(row[j])

    # 通过匹配节点名称来查找或创建节点
    node_objs = []  # 存储匹配或创建的节点对象
    for node_name in nodes:
        node = matcher.match("Node", name=node_name).first()
        if not node:
            node = Node("Node", name=node_name)
            graph.create(node)
        node_objs.append(node)

    # 创建关系（假设关系的数量总是比节点数量少一个）
    for k in range(len(relations)):
        relationship = Relationship(node_objs[k], relations[k], node_objs[k + 1])
        graph.create(relationship)

if __name__ == '__main__':
    path = r"C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\data\Graph_Data_ALL（1）.csv"
