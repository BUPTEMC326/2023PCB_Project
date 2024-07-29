from py2neo import Graph, NodeMatcher, Node, Relationship
import csv

# 连接到Neo4j数据库
graph = Graph("http://localhost:7474", auth=("neo4j", "123456"))

# 删除所有现有数据（小心使用）
graph.delete_all()

# 读取CSV文件
with open(r'D:\wk\Graph_Data_ALL.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)

# 打印第一行数据以验证读取成功
print(data[1])

# 创建节点和关系
matcher = NodeMatcher(graph)

for i in range(1, len(data)):
    head_name = data[i][1]
    relation = data[i][2]
    tail_name = data[i][3]

    # 查找或创建头节点
    head_node = matcher.match("head", name=head_name).first()
    if not head_node:
        head_node = Node("head", name=head_name)
        graph.create(head_node)

    # 查找或创建尾节点
    tail_node = matcher.match("tail", name=tail_name).first()
    if not tail_node:
        tail_node = Node("tail", name=tail_name)
        graph.create(tail_node)

    # 创建关系
    relationship = Relationship(head_node, relation, tail_node)
    graph.create(relationship)




if __name__ == '__main__':
     path = r"D:\wk\Graph_Data_ALL.csv"
     # call apoc.export.csv.query("MATCH (n)-[r]-(m) return n,r,m","d:/movie.csv",{format:'plain',cypherFormat:'updateStructure'})
