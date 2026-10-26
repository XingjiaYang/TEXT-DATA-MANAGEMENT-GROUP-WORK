# Yelp 评论搜索引擎（AI6122 作业 · 3.3 搜索引擎）

基于 Yelp Open Dataset 的评论搜索引擎。倒排索引、TF-IDF 余弦相似度和 Okapi BM25 均**从零实现**，并提供：

- **网页界面**（FastAPI）：BM25 / TF-IDF 切换、三栏对比、k1 和 b 滑块实时调参、每条结果的得分明细
- **命令行查询**：单次查询或交互模式
- **短语查询**：基于位置索引，`"not good"` 只匹配两个词相邻出现的评论
- **Elasticsearch 对照（可选，默认关闭）**：用 ES 默认配置索引同一批评论，在网页上与自己的实现并排比较

---

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux（在 Ubuntu 26.04 上测试） |
| Python | 3.12，由 [uv](https://docs.astral.sh/uv/) 管理，**不需要 Conda** |
| 内存 | 全量建索引时峰值约 33 GB；查询服务本身很小（索引用 mmap 加载） |
| 磁盘 | 原始数据约 9 GB；生成文件约 23 GB（见下表） |
| Docker | 仅 Elasticsearch 对照需要 |

生成文件的大小：

| 路径 | 内容 | 大小 |
|---|---|---|
| `docs/` | 每条评论一个 Markdown 文件（697 万个） | 约 17 GB |
| `catalog.sqlite` | 文档元数据目录 | 约 2 GB |
| `index/` | 倒排索引（numpy 数组） | 约 3.3 GB |
| ES 数据卷 `es-data` | Elasticsearch 索引（可选） | 约 5.6 GB |

---

## 2. 快速开始

所有命令都在项目根目录下执行。

### 2.1 安装依赖

```bash
# 安装 uv（已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 按 pyproject.toml / uv.lock 创建 .venv 并安装依赖
uv sync
```

依赖：`numpy`、`nltk`、`fastapi`、`uvicorn`、`elasticsearch`（仅 ES 对照使用）。分词器**不需要下载任何 NLTK 数据**。

### 2.2 准备数据

从 <https://www.yelp.com/dataset> 下载数据集，解压后把以下两个文件放到 `data/yelp/`：

```
data/yelp/yelp_academic_dataset_business.json
data/yelp/yelp_academic_dataset_review.json
```

### 2.3 预处理：清洗、去重、抽取成 Markdown（约 4 分钟）

```bash
uv run python -m app.preprocess.extract_reviews \
    --reviews data/yelp/yelp_academic_dataset_review.json \
    --business data/yelp/yelp_academic_dataset_business.json \
    --out docs --db catalog.sqlite
```

- 清洗：HTML 反转义、Unicode NFC 规范化、换行统一为 LF、去掉控制字符
- 去重：空白归一化后文本完全相同的评论只保留最早的一条（共去掉 16,607 条）
- 输出：`docs/<business_id>_<商家名>/<review_id>.md`，YAML front matter 里是商家和评论的元数据，正文只有评论文本
- `catalog.sqlite`：`reviews`（`doc_id` ↔ 路径、星级、日期等）、`businesses`、`dropped`（被去掉的评论及原因）、`meta`

### 2.4 建倒排索引（约 3 分钟）

```bash
uv run python -m app.search.indexer --catalog catalog.sqlite --out index
```

想先在小数据上试，可以加 `--limit 30000 --out index_small`。重建时加 `--force`。

### 2.5 启动网页界面

```bash
uv run python -m app.search.service
```

浏览器打开 <http://127.0.0.1:8000>（服务默认监听 `0.0.0.0:8000`，同一网络的其他机器可以用本机 IP 访问）。

- 只想本机访问：`--host 127.0.0.1`
- 换端口：`--port 8080`
- 第一次启动会花约 8 秒计算 TF-IDF 文档范数，并缓存到 `index/tfidf_norm.npy`，之后启动约 0.1 秒

### 2.6 命令行查询

```bash
# 单次查询；--model 可选 bm25 / tfidf / both（两者都跑）
uv run python -m app.search.ranker "authentic carbonara" --model both -k 5

# 修改 BM25 参数
uv run python -m app.search.ranker "vegan friendly brunch" --k1 1.5 --b 0.5

# 交互模式（索引只加载一次）
uv run python -m app.search.ranker
#   :model tfidf    切换模型（bm25 / tfidf / es / both / all）
#   :k1 1.5  :b 0.3 修改 BM25 参数
#   :k 20           每个模型返回的条数
#   :q              退出
```

---

## 3. 查询语法

| 写法 | 含义 |
|---|---|
| `authentic carbonara` | **OR 语义**（和普通搜索引擎一样）：包含任意一个词就参与打分 |
| `"happy hour" cheap drinks` | 引号内是**短语**：结果必须包含该短语，其余词照常参与打分 |
| `"food was good"` | 停用词保留位置：只匹配 food 和 good 之间正好隔一个词的写法 |

文档和查询使用**同一套分词流程**，所以 `Waiting` 能匹配 `waited`，`don't` 会变成 `do not`。

---

## 4. Elasticsearch 对照（可选，默认关闭）

不启用时，网页和命令行都只有自己的 BM25 和 TF-IDF，不需要 Docker。

### 4.1 启动 ES 并导入数据（导入约 1 分钟）

```bash
docker compose up -d                       # 启动 ES 9.5.5（见 compose.yaml）
curl http://127.0.0.1:9200                 # 能返回版本信息即启动成功
uv run python -m app.search.es index       # 导入全部评论；小规模测试可加 --limit 30000
```

- ES 只绑定本机 `127.0.0.1:9200`，并关闭了安全认证（本机实验用）
- 堆内存默认 8 GB，可以这样调整：`ES_HEAP=4g docker compose up -d`
- 重新导入：`uv run python -m app.search.es index --force`

### 4.2 启用 ES 对照

```bash
uv run python -m app.search.service --es            # 网页：多出 Elasticsearch 按钮，Compare 变成三栏
uv run python -m app.search.ranker "pizza" --es --model all   # 命令行：三个模型都跑
uv run python -m app.search.es search "pizza"       # 只查 ES
```

**必须显式加 `--es` 才会启用**。如果加了 `--es` 但 ES 没有启动或还没导入数据，服务会打印提示并自动关闭 ES 对照，其余功能照常运行。

### 4.3 ES 的设置

ES 使用**默认的 `standard` 分析器**（只做切词和转小写，不去停用词、不做词干化）和默认 BM25（k1 = 1.2，b = 0.75），**分词不做任何修改**。只改了以下几项与分词、打分无关的设置：

| 设置 | 原因 |
|---|---|
| `number_of_shards: 1` | 让 df 和 avgdl 是全局统计，和我们的索引一致 |
| `_id = doc_id` | 两边的结果可以按同一篇文档比对 |
| `_source` 关闭 | 评论原文从 `docs/` 读取，和我们的引擎一样，节省空间 |

所以这是**端到端的系统对比**：差异同时来自分词（词干化、停用词、标点处理）和打分实现。网页上 ES 的 k1 和 b 固定为默认值，滑块只影响我们自己的 BM25。

### 4.4 停止 ES

```bash
docker compose down        # 停止，保留索引数据
docker compose down -v     # 停止并删除索引数据
```

---

## 5. 设计说明

### 5.1 分词（`app/search/tokenizer.py`）

基于 NLTK `TreebankWordTokenizer`，流程参照 Elasticsearch 的 `english` 分析器：

1. **规范化**：弯引号、反引号、`´` 统一为 `'`；删除 URL；把连续的点（`good..but`）、省略号、零宽字符、全角和 CJK 标点、句号后漏掉的空格（`carbonara.First`）换成空格；拆开用 `- / \ _ =` 连接的词（`the-best-food`）
2. **切词**：Treebank
3. **清理**：转小写，去掉词两端的标点，`n't` → `not`，去掉 `'s` `'re` 等附着词，丢弃标点和 emoji
4. **停用词**：Lucene 英文停用词表，**但保留 `no` 和 `not`**，否则 "not good" 会变成 "good"
5. **词干化**：Porter（原始算法）

停用词删掉后**保留位置**，这样短语查询仍然精确。上述规范化让词表从 68.2 万降到 54.5 万。

### 5.2 倒排索引（`app/search/indexer.py`、`app/search/index.py`）

所有倒排表首尾相连存成 numpy 数组（CSR 格式），加载时用 mmap：

| 文件 | 类型 | 含义 |
|---|---|---|
| `vocab.txt` | 文本 | 一行一个词项，行号就是 term_id |
| `term_ptr.npy` | int64[V+1] | 词项 t 的倒排表位于 `post_docs[term_ptr[t]:term_ptr[t+1]]` |
| `post_docs.npy` | int32[P] | doc_id，同一个词项内升序 |
| `post_tf.npy` | uint16[P] | 词频 tf |
| `term_pos_ptr.npy` | int64[V+1] | 词项 t 的位置从 `positions[term_pos_ptr[t]]` 开始 |
| `positions.npy` | uint16[L] | 出现位置，每个倒排项占 tf 个 |
| `doc_len.npy` | int32[N+1] | 文档长度（分词后的词项数） |
| `meta.json` | JSON | N、avgdl（建索引时全局计算）等 |

建索引分两步：36 个进程各自为一段 doc_id 生成局部索引；父进程合并词表，对全部倒排项做稳定排序，再用向量化的方式重排位置数组。

### 5.3 排序（`app/search/ranker.py`）

**TF-IDF 余弦（SMART `ltc.ltc`）**

$$w_{t,d} = (1+\ln tf_{t,d})\cdot\ln\frac{N}{df_t},\qquad \text{score}(q,d)=\frac{\sum_t w_{t,q}\,w_{t,d}}{\lVert q\rVert\,\lVert d\rVert}$$

文档范数 ‖d‖ 对文档里的**全部**词项计算。

**Okapi BM25**

$$\text{score}(q,d)=\sum_{t\in q} qtf_t\cdot \text{IDF}(t)\cdot\frac{tf_{t,d}\,(k_1+1)}{tf_{t,d}+k_1\left(1-b+b\,\frac{|d|}{\text{avgdl}}\right)},\qquad \text{IDF}(t)=\ln\left(1+\frac{N-df_t+0.5}{df_t+0.5}\right)$$

IDF 用 Lucene 的非负版本：`not` 的 df（402 万）超过 N/2，用经典公式会得到负的 IDF。

打分方式是按词项累加：对每个查询词，用 numpy 把得分一次性加到它倒排表里的所有文档上，再用 `argpartition` 取前 k 个。

**正确性验证**：在 3 万篇文档上，用逐篇计算的暴力实现对照，两个模型在多组 k1、b 下的命中数、top-10 得分和每个词的贡献值全部一致。

---

## 6. API

| 方法与路径 | 参数 | 说明 |
|---|---|---|
| `GET /` | | 网页界面 |
| `GET /api/stats` | | 索引统计，以及 `es_enabled` |
| `GET /api/search` | `q`, `model`=bm25/tfidf/es, `k`, `offset`, `k1`, `b` | 单个模型的结果 |
| `GET /api/compare` | `q`, `k`, `k1`, `b` | 所有已启用模型的结果，以及两两之间的 overlap 和 Jaccard |

交互式接口文档：<http://127.0.0.1:8000/docs>。

---

## 7. 项目结构

```
app/
  preprocess/extract_reviews.py   清洗、去重，抽取成 Markdown，生成 catalog.sqlite
  search/tokenizer.py             分词（文档和查询共用）
  search/indexer.py               建倒排索引
  search/index.py                 读取索引（mmap），短语匹配
  search/ranker.py                TF-IDF / BM25、SearchEngine、命令行
  search/service.py               FastAPI 服务
  search/es.py                    Elasticsearch 对照：导入与查询
  web/index.html                  网页界面
compose.yaml                      Elasticsearch（Docker）
pyproject.toml / uv.lock          依赖
```

---

## 8. 实测数据（36 核、247 GB 内存）

| 步骤 | 耗时 |
|---|---|
| 预处理（697 万条评论 → Markdown + SQLite） | 3 分 35 秒 |
| 建倒排索引 | 3 分 04 秒（峰值内存 33 GB） |
| 首次计算 TF-IDF 范数 | 约 8 秒（之后读缓存） |
| ES 导入 | 65 秒 |
| 服务启动 | 约 0.1 秒 |
| 单次查询 | 30–220 ms（高频词更慢） |

索引规模：6,973,673 篇文档，545,227 个词项，3.89 亿个倒排项，5.1 亿个位置，avgdl = 73.36。

---

## 9. 常见问题

- **改了分词器之后**：必须重建索引（`indexer --force`），因为文档和查询要用同一套分词。TF-IDF 范数缓存放在索引目录里，会随索引一起重建。
- **网页显示 "Search failed"**：确认服务还在运行，并检查 `logs/service.log`。
- **加了 `--es`，网页上却没有 Elasticsearch 按钮**：ES 没有启动或还没导入数据，服务启动时会打印 `Elasticsearch disabled: ...`。按第 4.1 节操作后重启服务。
- **内存不够建全量索引**：用 `--limit` 只索引一部分文档。
