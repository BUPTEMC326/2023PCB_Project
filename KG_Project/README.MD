## 基于知识图谱的PCB干扰诊断和辐射预测


1每个元件是一个class类，所有的变量（比如时钟工作频率）放在一起，集中处理
2对于IO类，采用继承机制，将公共属性和方法定义在父类里，HDMI这些子类实现各自特有的方法。
3为了减少查询节点上的函数代码，添加config模块，简化cql查询过程,采用层级递进（IO->HDMI），减少代码冗余
4实例化对象时 对输入添加异常处理，要求只能输入指定类型的数据，否则进行异常处理
5实现了辐射发射四个模型 辐射计算
6 IO 模块的规则实现了
7 目前，实现一条规则：需要动的地方：对于添加节点：查看建立节点模块是否有该节点，没有的话，在config配置，配置时需要考虑是否
有集合节点，并实例化类以创建对象用于规则判断，有的话直接添加属性，实现属性函数即可。
对于给节点添加属性，只需要在config里给对应节点加属性即可


### 环境

- python    >= 3.6
- pyahocorasick==1.4.2
- requests==2.25.1
- gevent==1.4.0
- jieba==0.42.1
- six==1.15.0
- gensim==3.8.3
- matplotlib==3.1.3
- Flask==1.1.1
- numpy==1.16.0
- bert4keras==0.9.1
- tensorflow==1.14.0
- Keras==2.3.1
- py2neo==2020.1.1
- tqdm==4.42.1
- pandas==1.0.1
- termcolor==1.1.0
- itchat==1.3.10
- ahocorasick==0.9
- flask_compress==1.9.0
- flask_cors==3.0.10
- flask_json==0.3.4
- GPUtil==1.4.0
- pyzmq==22.0.3
- scikit_learn==0.24.1

### 效果展示



### 如何运行demo代码



1. **启动neo4j**

打开cmd ，进入安装目录，输入`neo4j.bat console`就行；如果是Linux系统，你可以输入`./neo4j start`后台启动

```


```
python PCBNeo4j.py
```


