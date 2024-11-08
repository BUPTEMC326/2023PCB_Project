from neo4j import GraphDatabase


class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_fpc_knowledge_graph(self):
        with self.driver.session() as session:
            # 实体-关系-实体
            session.execute_write(self._create_fpc_triples)
            # 实体-属性-属性值
            session.execute_write(self._create_fpc_properties)

    @staticmethod
    def _create_fpc_triples(tx):
        # 创建4层FPC板的三元组关系
        triples = [
            ("4层FPC板", "包含", "BTB连接器"),
            ("4层FPC板", "包含", "grounding"),
            ("4层FPC板", "包含", "中间位置"),
            ("grounding", "包含", "地孔")
        ]
        for entity1, relation, entity2 in triples:
            query = (
                f"MERGE (a:Entity {{name: $entity1}}) "
                f"MERGE (b:Entity {{name: $entity2}}) "
                f"MERGE (a)-[:{relation}]->(b)"
            )
            tx.run(query, entity1=entity1, entity2=entity2)

    @staticmethod
    def _create_fpc_properties(tx):
        # BTB连接器和地孔设置属性
        properties = [
            ("BTB连接器", "设计长度", 40),  # 单位mm
            ("地孔", "数目", 100)  # 单位个
        ]
        for entity, prop, value in properties:
            query = (
                f"MERGE (e:Entity {{name: $entity}}) "
                f"SET e.{prop} = $value"
            )
            tx.run(query, entity=entity, value=value)


# 连接到Neo4j数据库并创建知识图谱
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

kg = KnowledgeGraph(uri, user, password)

# 创建4层FPC板的知识图谱
kg.create_fpc_knowledge_graph()

kg.close()
