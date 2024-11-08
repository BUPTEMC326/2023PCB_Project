import re

from py2neo import Graph
import json

graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))


# cql知识图谱获取维度
def get_MIPI_C_dimension():
    cql_query = """
    MATCH (阻抗控制线 {name: 'lane'})-[:维度]->(d)
    RETURN d
    """
    result = graph.run(cql_query).data()
    dimension = re.search(r"name: '(\[.*?\])'", str(result[0]['d'])).group(1)
    return json.loads(dimension)

# 判断
def check_MIPI_C_dimension(extracted_data_path):
    # 板子数据与知识图谱数据
    with open(extracted_data_path, 'r', encoding='utf-8') as file:
        extracted_data = json.load(file)
    standard_dimension = get_MIPI_C_dimension()
    extracted_dimension = extracted_data.get('dimension')

    all_compliant = True
    for i, (extracted_value, standard_value) in enumerate(zip(extracted_dimension, standard_dimension)):
        if extracted_value != standard_value:
            lane_name = f"lane{i-1}"
            if i == 0:
                print(f"FPC lane组数为{extracted_value}，不合规范，应包含{standard_value}组lane")
            else:
                print(f"第{i-1}组lane包含{extracted_value}线，不合规范，应包含{standard_value}线")
                for component in extracted_data[lane_name]:
                    print(f"  - {component}")
            all_compliant = False

    if all_compliant:
        print("三线差分设计符合规范")
    return


extracted_data_path = 'extractedData/MIPI_C_PHY.json'
check_MIPI_C_dimension(extracted_data_path)

