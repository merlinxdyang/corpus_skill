# English Corpus Prep Skill

这个 Skill 用于把**多种格式的英文原始文件**整理成可直接用于 NLP/语料分析的 `TXT` 语料。

## 仓库结构（用于 GitHub）

```text
corpus_skill/
├── README.md
├── .gitignore
├── skills/
│   └── public/
│       └── english-corpus-prep/
│           ├── SKILL.md
│           ├── agents/openai.yaml
│           ├── scripts/build_corpus.py
│           └── references/cleaning-profile.md
└── examples/
    └── sample-output-two-pdfs/
```

## 功能概览

- 自动检测输入文件格式（先检测再抽取）
- 按格式提取文本内容
- 统一清洗与规范化文本
- 统一输出 UTF-8 编码文本
- 自动记录错误日志并跳过异常文件
- 自动提取并单独保存每个文档的元信息
- 生成 PTB（Penn Treebank）词性标注模板语料
- 输出语料统计报告（如 token/type）

## 支持输入格式

- PDF（`*.pdf`）
- 纯文本/Markdown（`*.txt`, `*.md` 等）
- HTML/XML（`*.html`, `*.htm`, `*.xml`）
- Word（`*.docx`）
- JSON/JSONL/NDJSON（`*.json`, `*.jsonl`, `*.ndjson`）
- CSV/TSV（`*.csv`, `*.tsv`）
- 其他文本类文件（回退为文本解码）

## 清洗规则（默认）

- Unicode 规范化（NFKC）
- 统一换行符
- 删除软连字符（`\u00ad`）
- 修复断行连字符（如 `hyphen-\nated -> hyphenated`）
- 合并多余空格与过多空行
- 过滤常见页码噪声（如 `Page 3`, `Page 3 of 10`, 纯数字行）

详细规则见：`skills/public/english-corpus-prep/references/cleaning-profile.md`

## 异常与日志机制

以下情况会被**记录日志并跳过该文件**：

1. 文档损坏、无法解析（例如 PDF/DOCX/XML/JSON 结构损坏）
2. 编码错误，无法转换为 UTF-8
3. 无法提取可读文本（例如扫描版 PDF 未 OCR）
4. 清洗后为空（可用于后续排查）

错误日志文件：

- `输出目录/logs/errors.jsonl`

每条日志包含：源文件路径、检测格式、错误码、错误信息、修复提示。

## 编码策略（UTF-8 统一）

- 所有输出文件统一为 `UTF-8`
- 输入文件若非 UTF-8，会先尝试转码再处理
- 若无法转码为 UTF-8：记录日志并跳过

## PDF 正文提取策略（先去元信息，再入库）

针对 PDF，会先做“正文筛选”，再进入语料清洗：

- 提取并移除封面/标题页元信息（单独保存）
- 提取并移除版权页（Copyright Information）信息（单独保存）
- 删除目录（Contents / Table of Contents）相关内容
- 删除参考文献尾部（References / Bibliography / Works Cited）内容
- 删除脚注样式行、重复页眉页脚、`This page is intentionally left blank.` 等噪声

然后才进行统一清洗和词性标注。

## 元信息输出

每个文件的元信息会单独保存，并从清洁语料文本中移除：

- `corpus_out/metadata/per_file/*_metadata.json`
- `corpus_out/metadata/metadata_manifest.jsonl`

若是 PDF，会额外记录：

- PDF 内嵌元数据（如 Creator/Producer/CreationDate）
- 页数
- 被移除的元信息块（如封面块、版权块）

## 大文件/大规模输入安全确认

- 当**单个文件超过 100MB**时，会请求确认是否继续
- 当**总数据量过大或文件数过多**时，会请求确认，并给出：
  - 估计处理时间
  - 模型调用费用提示（本脚本本地处理无模型费用；若后续把语料送入大模型，会给出大致 token 规模）

若在非交互环境执行，可加 `--assume-yes` 跳过确认。

## 使用方法

在仓库根目录运行：

```bash
python3 skills/public/english-corpus-prep/scripts/build_corpus.py <输入路径1> [<输入路径2> ...] --output-dir <输出目录> [--recursive] [--skip-empty] [--assume-yes]
```

示例 1：处理整个目录（递归）

```bash
python3 skills/public/english-corpus-prep/scripts/build_corpus.py ./raw_data --output-dir ./corpus_out --recursive
```

示例 2：处理多个指定文件

```bash
python3 skills/public/english-corpus-prep/scripts/build_corpus.py ./raw/a.pdf ./raw/b.html ./raw/c.json --output-dir ./corpus_out
```

示例 3：非交互环境下处理超大数据（自动确认）

```bash
python3 skills/public/english-corpus-prep/scripts/build_corpus.py ./raw_data --output-dir ./corpus_out --recursive --assume-yes
```

## 输出结构

设 `--output-dir corpus_out`，会生成：

- `corpus_out/cleaned_corpus/cleaned_txt/*.txt`：每个源文件对应一个清洗后文本
- `corpus_out/cleaned_corpus/corpus.txt`：合并后的清洁总语料
- `corpus_out/cleaned_corpus/manifest.jsonl`：每条记录包含
  - `source`（源文件绝对路径）
  - `detected_format`（检测格式）
  - `output_file`（输出文件绝对路径）
  - `chars_raw` / `chars_clean`（清洗前后字符数）
  - `source_encoding` / `converted_to_utf8`（编码与转码信息）

- `corpus_out/pos_annotated_corpus/annotated_txt/*.txt`：逐文件词性标注模板
- `corpus_out/pos_annotated_corpus/corpus_pos_template.txt`：合并后的词性标注模板语料
- `corpus_out/pos_annotated_corpus/manifest_pos.jsonl`：标注文件索引
- `corpus_out/logs/errors.jsonl`：错误日志
- `corpus_out/metadata/per_file/*_metadata.json`：每个文件的元信息
- `corpus_out/metadata/metadata_manifest.jsonl`：元信息清单
- `corpus_out/corpus_report.json`：语料统计报告（包括 token/type、处理量、跳过量、格式分布等）

## PTB 词性标注模板格式

标注模板使用 Penn Treebank 风格，格式为：

- `单词_词性`
- `标点_词性`

示例：

```text
This_DT is_VBZ a_DT sample_NN ._.
```

标点也会被标注（例如 `,_ ,`、`._.`、`(_-LRB-`、`)_-RRB-`）。

## 依赖说明

- 必需：Python 3
- PDF 抽取建议安装其一：
  - `pypdf`（优先）
  - `pdfplumber`（回退）

当 PDF 依赖未安装时，非 PDF 文件仍可正常处理。

## 质量检查建议

每次运行后建议检查：

1. `cleaned_corpus/manifest.jsonl` 是否记录了编码与格式信息
2. `logs/errors.jsonl` 中的错误是否可修复（编码、损坏、OCR 缺失等）
3. 抽样检查 `cleaned_txt` 与 `annotated_txt` 是否符合预期
4. 查看 `corpus_report.json` 的 token/type、跳过文件数、格式分布
