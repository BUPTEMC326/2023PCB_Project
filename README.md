1.构建图谱：
- buildMap.py

2.阻抗控制线：
- 提取阻抗控制线：getImpedenceControl.py
  - 提取阻抗控制线维度：class MIPI
  - 提取三线差分：class MIPI_C_PHY
  - 提取双线差分：class MIPI_D_PHY

- 三线差分cql：MIPI_C_PHY.py
  - cql知识图谱：def get_MIPI_C_dimension, def get_MIPI_C_PHY_width
  - 判断：def check_MIPI_C dimension, def check_MIPI_C_PHY_width

- 双线差分cql：MIPI_D_PHY.py
  - cql知识图谱：def get_MIPI_D_dimension, def get_MIPI_D_PHY_width
  - 判断：def check_MIPI_D_dimension, def check_MIPI_D_PHY_width

- 检查阻抗控制线规则：checkImpedence.py

3.Grounding：
- 提取地孔：getGrounding.py
  - 提取地孔数目：class grounding_Num
  - 提取地孔距离：class grounding_Dis

代码上传到github里大家可以实时更新
