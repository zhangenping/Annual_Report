# 企业级年报查询系统

本仓库当前处于 **RAG 知识库优先** 阶段。`docs/report/` 下两份 PDF 年报可作为首批入库样本。

## 文档

| 文档 | 说明 |
|------|------|
| [docs/01-注意事项.md](docs/01-注意事项.md) | 建设知识库注意事项 |
| [docs/02-技术细节与规划.md](docs/02-技术细节与规划.md) | 技术架构与实施路线 |

## 项目结构

```
annual_report_rag/     # 核心 Python 包
├── pipelines/         # 解析、切片、索引、入库
├── retrieval/         # 混合检索 + Rerank
├── agents/            # LangGraph Agent + 工具
├── api/               # FastAPI 接口
├── schemas/           # Pydantic 数据模型
└── storage/           # 本地存储抽象
configs/               # pipelines.yaml / models.yaml
scripts/               # CLI 脚本
eval/                  # 评估数据集与脚本
docs/report/           # 原始年报 PDF
data/                  # 解析产物与索引（运行后生成）
```

## 快速开始

### 1. 安装依赖

```bash
cd "f:\Annual Report"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

首次运行会自动下载 Embedding 模型（`BAAI/bge-small-zh-v1.5`），体积约数百 MB。

### 2. 批量入库（解析两份年报 + 建索引）

```bash
python scripts/batch_ingest.py
```

### 3. 命令行检索

```bash
python scripts/search_cli.py "营业收入" --top-k 5
```

### 4. 启动 API

```bash
uvicorn annual_report_rag.api.app:app --reload --port 8000
```

- `POST /api/v1/search` — 混合检索
- `POST /api/v1/ask` — Agent 问答（需配置 `OPENAI_API_KEY`）

### 5. 可选：Agent 环境变量

复制 `.env.example` 为 `.env` 并填写：

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

## 解析引擎

默认 `auto`：已安装 [Docling](https://github.com/DS4SD/docling) 时优先使用，否则回退 **PyMuPDF + pdfplumber**（表格与图片提取）。

```bash
pip install docling   # 可选增强
```

在 `configs/pipelines.yaml` 中可设置 `parse.engine: pymupdf | docling | auto`。

## 样本数据

当前 `docs/report/` 包含：

- `H2_AN202604031821016965_1.pdf`
- `H2_AN202604281821703679_1.pdf`

入库后可在 `data/parsed/`、`data/chunks/`、`data/index/` 查看中间产物。
