import csv
from py2neo import Graph, Node, Relationship, NodeMatcher

# 连接neo4j数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

# 清空数据库，慎用
graph.delete_all()

# 创建函数以确保节点存在
def get_or_create_node(label, **properties):
    matcher = NodeMatcher(graph)
    node = matcher.match(label, **properties).first()
    if node:
        return node
    else:
        node = Node(label, **properties)
        graph.create(node)
        return node

# 标签映射（四层fpc板包含的组件）
label_mapping = {
    "四层fpc板":"FPC_4",
    "阻抗控制线": "Impedence Control",
    "BTB": "BTB",
    "grounding": "grounding",
    "射频耦合线": "RF Coupling Line",
}

# 读取board文件
csv_file_path2 = r"C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\mapRule\board1.csv"
with open(csv_file_path2, 'r', encoding='ANSI') as f:
    reader = csv.DictReader(f)
    for row in reader:
        head = row['head']
        relation = row['relation']
        tail = row['tail']

        # 标签映射
        head_label = label_mapping.get(head, "Component")
        tail_label = label_mapping.get(tail, "Component")

        head_node = get_or_create_node(head_label, name=head)
        tail_node = get_or_create_node(tail_label, name=tail)

        graph.create(Relationship(head_node, relation, tail_node))

# 读取阻抗控制线文件
csv_file_path1 = r"C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\mapRule\impedenceControl2.csv"
with open(csv_file_path1, 'r', encoding='ANSI') as f:
    reader = csv.DictReader(f)
    for row in reader:
        head = row['head']
        relation = row['relation']
        tail = row['tail']

        head_node = get_or_create_node("Impedence Control", name=head)
        tail_node = get_or_create_node("Impedence Control", name=tail)

        graph.create(Relationship(head_node, relation, tail_node))


print("导入成功。")
