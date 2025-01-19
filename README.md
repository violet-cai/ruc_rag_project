# RUC_RAG_PROJECT

## 项目架构

```bash
.
├── data
│   └── 存储知识库数据
│—— dataprocess
│   └── 数据处理相关代码
├── examples
│   └── results
│       └── 提供使用示例及结果展示
└── rag
    ├── config
    │   └── 配置文件目录
    ├── database
    │   └── 数据库相关代码
    ├── evaluator
    │   └── 评估相关代码
    ├── generator
    │   └── 生成相关代码
    ├── reranker
    │   └── 重排相关代码
    └── retriever
        └── 检索相关代码
```

## 安裝指南

在开始之前，请确保已根据 `requirements.txt` 文件安装所有必要的依赖包。你可以使用以下命令进行安装：

```bash
pip install -r requirements.txt
```

## 配置说明

根据自身设备配置修改 `rag/config/basic_config.yaml` 文件中的相关参数，以确保系统正常运行。

### `db_uri`

`db_uri` 指定了 Milvus 数据库的连接地址。根据你的环境，选择合适的配置：

- 如果你希望将数据存储在本地文件中，请使用以下配置：
  ```yaml
  db_uri: "./milvus.db" # 存储至本地milvus.db文件中
  ```
- 如果你已经将 Milvus 部署在 Docker 容器中，并希望连接该容器中的数据库，请使用以下配置：
  ```yaml
  db_uri: "http://localhost:19530" # 连接docker中数据库
  ```

### `db_dense_index_type`

`db_dense_index_type` 指定了向量索引的类型。不同的环境支持不同的索引类型：

- 在本地环境下，推荐使用 `HNSW` 索引：
  ```yaml
  db_dense_index_type: "HNSW" # 本地环境下不允许IVF_FLAT索引
  ```
- 在 Docker 环境中，可以使用 `IVF_FLAT` 索引以获得更快的检索速度：
  ```yaml
  db_dense_index_type: "IVF_FLAT" # docker中可以使用该索引更快
  ```


## 使用示例

### 创建数据库

执行以下命令使用部分数据生成数据库：

```bash
python examples/database_demo.py
```

### 执行检索增强

执行以下命令进行检索增强生成：

```bash
python examples/retrieve_demo.py
```

### 使用服务端-客户端方式运行

1. **服务端和客户端单独运行**:

首先执行下列命令开启服务端：

```bash
python examples/server_demo.py
```

随后新建一个终端，在新终端窗口中执行下列命令开启服务端：

```bash
python examples/client_demo.py
```


2. **脚本一键运行**:

`run_server_client.py`脚本中整合了运行服务端与客户端的代码，仅需执行下列命令即可一键运行服务：

```bash
python examples/run_server_client.py
```


示例代码中已经存放了一定的历史对话内容，用户可前往`server_demo.py`中查看，客户端开启后用户可在终端窗口输入query进行检索增强生成。


### 自定义修改

用户可以参考 examples 目录下的示例代码，根据自身需求进行修改和扩展。例如，可以自定义数据生成逻辑、调整索引参数或优化检索策略等。

### 注意事项

- 确保 Milvus 服务已启动并可正常连接。
- 根据实际数据量和查询需求，合理选择索引类型和参数。
- 在进行大规模数据操作时，注意资源消耗和性能优化。
