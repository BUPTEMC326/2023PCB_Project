import csv
from py2neo import Graph, Node, Relationship, NodeMatcher

# 连接Neo4j
try:
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
except Exception as e:
    print(f"建立知识图谱时出错: {e}")

graph.delete_all()

# 4层FPC节点
try:
    with open('mapTest/fpc_board.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            if reader.line_num == 1:
                continue
            fpc_node = Node("FPC_Board_4", id=line[0], name=line[1], layer=line[2])
            graph.merge(fpc_node, "FPC_Board_4", "id")
except Exception as e:
    print(f"建立FPC节点时时出错: {e}")


# btb_length_standard = 40.0

# ground_holes_standard = 100
#
# btb_node = Node("BTB_Connector", length=btb_length_standard)
# grounding_node = Node("Grounding", ground_holes=ground_holes_standard)
# graph.merge(btb_node, "BTB_Connector", "length")
# graph.merge(grounding_node, "Grounding", "ground_holes")

# BTB连接器节点，固定长度为 40mm
try:
    with open('mapTest/btb_connector.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            if reader.line_num == 1:  # 跳过表头
                continue
            btb_node = Node("BTB_Connector", id=line[0], length=line[1])
            graph.merge(btb_node, "BTB_Connector", "id")
except Exception as e:
    print(f"建立BTB节点时出错: {e}")

# 连接FPC与BTB
try:
    with open('mapTest/btb_to_fpc.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        matcher = NodeMatcher(graph)
        for item in reader:
            if reader.line_num == 1:
                continue
            fpc_node = matcher.match("FPC_Board", id=item[0]).first()
            btb_node = matcher.match("BTB_Connector", id=item[1]).first()
            if fpc_node and btb_node:
                relationship = Relationship(fpc_node, "包含", btb_node)
                graph.merge(relationship)
except Exception as e:
    print(f"连接FPC与BTB时出错: {e}")


# 连接FPC与Grounding
try:
    with open('mapTest/grounding_to_fpc.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for item in reader:
            if reader.line_num == 1:
                continue
            fpc_node = matcher.match("FPC_Board", id=item[0]).first()
            grounding_node = matcher.match("Grounding", id=item[1]).first()
            if fpc_node and grounding_node:
                relationship = Relationship(fpc_node, "包含", grounding_node)
                graph.merge(relationship)
except Exception as e:
    print(f"连接FPC与Grounding时出错: {e}")


# Grounding节点，固定地孔数为 100
try:
    with open('mapTest/grounding.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            if reader.line_num == 1:  # 跳过表头
                continue
            grounding_node = Node("Grounding", id=line[0], ground_holes=line[1])
            graph.merge(grounding_node, "Grounding", "id")
except Exception as e:
    print(f"建立grounding节点时出错: {e}")


# 判断BTB连接器长度是否符合规则
def validate_btb_length(new_btb_file):
    with open(new_btb_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        matcher = NodeMatcher(graph)
        for line in reader:
            if reader.line_num == 1:  # 跳过表头
                continue
            btb_id = line[0]
            new_length = line[1]
            btb_node = matcher.match("BTB_Connector", id=btb_id).first()

            if btb_node:
                btb_length = btb_node["length"]
                #print(f"知识图谱中BTB连接器{btb_id}的长度为{btb_length}mm")
                if new_length > btb_length:
                    print(f"BTB连接器{btb_id}的长度为{new_length}mm，应小于{btb_length}mm。")
                else:
                    print(f"BTB 连接器 {btb_id}的长度为 {new_length}mm，符合规则。")
            else:
                print(f"Error: BTB连接器{btb_id}不存在。")


# 判断地孔数是否符合规则
def validate_ground_holes(new_grounding_file):
    with open(new_grounding_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        matcher = NodeMatcher(graph)
        for line in reader:
            if reader.line_num == 1:  # 跳过表头
                continue
            grounding_id = line[0]
            new_ground_holes = line[1]
            grounding_node = matcher.match("Grounding", id=grounding_id).first()

            if grounding_node:
                ground_holes = grounding_node["ground_holes"]
                #print(f"知识图谱中 Grounding {grounding_id} 的地孔数为 {ground_holes}")
                if new_ground_holes < ground_holes:
                    print(f"Grounding{grounding_id}的地孔数为{new_ground_holes}，应大于{ground_holes}。")
                else:
                    print(f"Grounding{grounding_id}的地孔数为{new_ground_holes}，符合规则。")
            else:
                print(f"Error: Grounding{grounding_id}不存在。")


# 验证
new_btb_file = 'mapTest/btb_data.csv'
new_grounding_file = 'mapTest/grounding_data.csv'

validate_btb_length(new_btb_file)
validate_ground_holes(new_grounding_file)
