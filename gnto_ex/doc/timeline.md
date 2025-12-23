
# Report

## 代码
- 代码阅读
目前的LIMAO使用的是之前Bao论文的代码框架,在他的基础上进行修改.主要改动的都是BaoServer的代码部分.
整体思想就是借用psql的扩展机制,和plan_hint这个扩展模块,来实现查询计划的推荐.

- 代码运行
目前的代码主要使用了两个数据库,一个是IMDB,一个是TPC-H,使用的版本就是psql12.5.
根据我在代码打log来观察整体的流程的输入输出,可以绘制出如下的流程图,来确保在接下来我要将我的模型集成到这个框架中时,不会出现大的问题.

```mermaid
sequenceDiagram
    autonumber
    %% 定义参与者
    participant Client as python启动程序<br>(run_queries.py)
    participant PG as PostgreSQL<br>(需安装Bao扩展)
    participant Server as Bao Server<br>(bao_server/main.py)
    participant Model as 神经网络<br>(BaoNet / Module Assigner)
    participant DB as 历史数据库<br>(bao_server/bao.db)

    Note over Client, DB: === 阶段一：当用户产生一条SQL时,进行查询推理与执行 ===

    Client->>PG: 1. 提交 SQL 查询
    activate PG
    
    PG->>PG: 生成一组候选计划 (Hints)
    PG->>Server: 2. 发送候选计划列表 (Port 9381)
    activate Server
    
    Note right of Server: LIMAO 核心逻辑<br>主要改动BaoServer的代码部分.
    Server->>Model: 3. 输入计划树特征
    activate Model
    Note right of Model: Module Assigner:<br>将树节点分配给专家模块<br>之后尝试替换为我的模型GNTONet.
    Model-->>Server: 预测每个计划的延迟
    deactivate Model
    
    Server-->>PG: 4. 返回最优计划的索引 (Best Plan)
    deactivate Server

    PG->>PG: 5. 按照推荐计划执行查询
    PG-->>Client: 6. 返回查询结果
    
    Note over Client, DB: === 阶段二：数据收集与模型更新 ===

    PG->>Server: 7. 发送真实执行时间
    activate Server
    Server->>DB: 8. 存储 <计划特征, 真实延迟>
    deactivate Server
    deactivate PG


    Note right of Client: 周期性或 Episode 结束时
    Client->>Server: 9. 调用 baoctl.py --retrain
    activate Server
    
    Server->>DB: 读取历史训练数据
    activate DB
    DB-->>Server: 返回数据
    deactivate DB

    Server->>Model: 10. 训练并更新权重 (Retrain)
    activate Model
    Model-->>Server: 模型更新完毕
    deactivate Model
    deactivate Server
```


## 文章阅读
阅读文章: **LIMAO: A Framework for Lifelong Modular Learned Query Optimization**
主要内容就是介绍了LIMAO的框架,以及在动态环境下的查询优化问题,和解决方式.比如在数据库的多次增删改过程中数据分布和负载发生变化,而别的模型就需要重新训练,但这个模型就不用.
(个人感觉是Online Learning的一种实现方式)
文章中解决难题是**灾难性遗忘**和**动态环境适应性差**

缺陷:
1. 目前的专家模型还是需要手动调控数量.
2. 计划处的切分也需要人为划分.
3. 基础模型相对简单.

## 个人文章优化

* **LIMAO痛点:**
    * LIMAO 成功提出了“模块化终身学习”的框架来解决灾难性遗忘问题。
    * 然而，LIMAO 的底层模块仍然依赖于传统的 **Tree-CNN** 或 **Tree-LSTM**。
    * **弱点：** Tree-CNN 假设固定的树形结构，难以捕捉长距离依赖（Long-range dependencies），且对复杂的算子交互（如复杂的 Join Order 或并行的子计划）表征能力有限。

* **我的解决方案:**
    * 1. 新模型使用: 提出一种基于 **GATv2** 的模块设计。
    * 2. 自定义新模块:**ResBlock** 用于解决GATv2的表达能力不足的问题.

## Baseline目标: 对比LIMAO
1. 强调**查询执行时间的节省 (Execution Time Gain)** 远远大于 **推理时间的增加 (Inference Overhead)**。
2.  **Ablation Study (消融实验):**
    * LIMAO + Tree-CNN
    * LIMAO + GATv2
    * LIMAO + GATv2 + ResBlock
