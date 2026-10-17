# From Communication Scale to Engagement Effectiveness

Reproducible code and archived results for the CGTN Español YouTube study.

本仓库提供 2025 年 CGTN Español YouTube 研究的核心工作流、整理后的已有数据和实验结果。直接使用 CSV 即可重算描述统计、互动结构、参与指标、组间比较、回归和敏感性分析，无需 API 密钥、原始 SQLite 数据库或重新调用模型。

仓库地址：<https://github.com/chensy-ruc/From-Communication-Scale-to-Engagement-Effectiveness>

## 快速复现

环境：Python 3.12–3.13，macOS 或 Linux；Windows 可使用 WSL。安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后执行：

```bash
git clone https://github.com/chensy-ruc/From-Communication-Scale-to-Engagement-Effectiveness.git
cd From-Communication-Scale-to-Engagement-Effectiveness
uv sync --locked --python 3.12
uv run --locked python -m cgtn.reproduce
```

默认输出为 `outputs/reproduction/`。程序先核验输入文件的 SHA-256，再重算结果并与 `data/reference/` 中已有结果逐表比较。查看 `outputs/reproduction/verification.json` 可获得每张表的核验状态；不匹配或缺失时命令以非零状态退出。

输出目录已存在时拒绝覆盖。再次运行可指定新目录，也可只检查数据完整性：

```bash
uv run --locked python -m cgtn.reproduce --output outputs/reproduction-2
uv run --locked python -m cgtn.reproduce --verify-only
```

依赖安装需要下载软件包；复现计算期间阻断网络，不读取 `.env`，不产生模型请求。参数取自 `data/study.toml`，固定种子 `20250917`、1,000 次 bootstrap；修改根目录 `config.toml` 不改变已有数据的复现设置。

## 目录与已有结果

```text
README.md          本说明：安装、复现、工作流与数据口径
src/cgtn/          采集、清洗、编码、统计及复现的核心代码
data/
  processed/       清洗数据、研究样本和采集覆盖
  inputs/          冻结编码输入、编码规则和初编敏感性视频表
  labels/          已有最终视频、评论标签
  checks/          检查选择清单与逐标签对照
  reference/       已有实验结果，作为复现比对基准
  metadata/        原分析参数、检查协议和处理摘要
  manifest.json    文件来源、行数、字段、字节数与 SHA-256
  study.toml       冻结的原研究配置
run.py             采集和编码工作流的命令入口
config.toml        新采集、编码实验的配置
.env.example       可选 API 凭证模板
pyproject.toml     Python 包和运行依赖
uv.lock            锁定的依赖版本
LICENSE            MIT 代码许可证
```

保留 **43 份 CSV**，与配置和元数据合计约 **164 MiB**。对应原运行 `cgtn_es_2025_snapshot_01`、编码实验 `agents_deepseek_economy_v1`。全部 51 个来源文件保留原始字节，CSV 的记录、标签、数值和顺序均未改写。

| 数据位置 | 内容及用途 |
|---|---|
| `data/processed/videos.csv` | 7,476 个清洗视频，描述统计和互动结构输入 |
| `data/processed/comments.csv` | 43,478 条清洗评论，保留原文、线程、时间和账户标识 |
| `data/processed/collection_coverage.csv` | 采集状态与数量，用于区分未采完、缺失与零评论 |
| `data/processed/sample.csv`、`sample_design.json` | 冻结的研究视频框和抽样设计 |
| `data/inputs/videos.csv`、`comments.csv`、`coverage.csv` | AI 编码时的冻结输入：7,476 个视频、43,477 条评论 |
| `data/inputs/initial_analysis_videos.csv` | 已有初编视频指标，供敏感性比较使用 |
| `data/inputs/codebook.txt` | 原编码规则；作为模型输入保留，未修改内容 |
| `data/labels/predictions_*.csv` | 最终标签、证据、来源及缺失/未决状态 |
| `data/checks/selection.csv`、`label_comparisons.csv` | 实际检查范围、抽样概率与逐标签初编/检查对照 |
| `data/reference/analysis/` | 语义指标、相关、组间比较、回归和敏感性结果 |
| `data/reference/structure/` | 线程与视频互动结构、相关、组间比较和回归 |
| `data/reference/descriptive/` | 视频、评论、语言、月份描述及排除记录 |
| `data/reference/checks/` | 重点与随机检查域的一致性指标 |
| `data/reference/audit/` | 6 张原执行摘要，保留为历史结果 |

清洗评论和编码输入相差一条，是因为受众编码框排除了一个频道自评根线程。`data/manifest.json` 的 `path` 指当前位置，`source` 指整理前的原路径。原始输入清单 `data/inputs/manifest.json` 内的 `codebook.md` 对应现有 `codebook.txt`，仅更改文件扩展名，原哈希保持不变。

## 复现范围与指标口径

本机已验证 **24 张重算结果表全部匹配，0 张不匹配**。比较检查行列结构、顺序、缺失位置和数值；数值容差为相对 `1e-7`、绝对 `1e-9`。另外 **7 张表仅作历史归档**：6 张执行审计汇总和 `excluded_videos.csv`，重建它们需要原始请求记录，不计入重算成功数。

| 重算内容 | 核心实现 |
|---|---|
| 视频、评论描述统计 | `analysis.py` |
| 互动结构与相应模型 | `structure.py` |
| 最终标签关联、语义指标与统计分析 | `ai_analysis.py` |
| 检查域一致性 | `ai_screening_report.py` |
| 初编与检查后结果的敏感性比较 | `reproduce.py` 复用原统计函数 |

`video_id`、`comment_id` 为平台记录标识，`root_id` 为根评论标识；`author_hash` 为已有假名化账户标识。CSV 仍含评论原文、平台 ID 和时间，假名化不意味着文本不可识别。每份 CSV 的完整字段和行数见 `data/manifest.json`。

- 视频标签分别描述 `topic`、`form`、`platform_format`。不能从时长直接断定 Shorts，也不能从标题推断未提供的字幕内容。
- 评论使用多标签编码，包括信息询问、事实核查、观点、有论据观点、跨文化比较、情绪、实质回应、学习/重新考虑表达、分享意图和垃圾信息。完整操作定义保存在 `data/inputs/codebook.txt`。
- `deep_rate` 为有效受众评论中认知性参与的比例；`mean_quality` 为 0–3 质量等级的辅助均值。垃圾评论和频道自评不进入受众质量分母。
- `dialogue_conversion`、`sustained_dialogue` 是线程内可观察互动的操作性代理。API 回复关系指向根评论，不能据此恢复精确的用户对用户回应网络。
- 缺失、未决和不可编码不当作零。学习表达不证明实际认知变化，分享意图不等于经平台后台验证的转发。

复现固定使用已有最终标签，不重新请求模型。初编敏感性从已有视频指标表开始，不能据此宣称已从原模型响应重新生成初编标签。

## 核心工作流与新实验

整体流程为：**采集 → 清洗与覆盖审计 → 冻结编码输入 → 全量 AI 初编 → 重点检查与非重点分层随机检查 → 指标与统计分析**。

| 工作步骤 | 源码模块 |
|---|---|
| 配置、SQLite、快照与账户假名化 | `core.py` |
| YouTube API、限速、配额、缓存与分页续传 | `api.py`、`collect.py` |
| 清洗、观察窗口、缺失与重复处理 | `clean.py` |
| 编码定义、人工编码与人工参考验证 | `coding.py`、`ai_eval.py` |
| 冻结输入、提示词、模型客户端 | `ai_data.py`、`ai_prompts.py`、`ai_client.py` |
| 编排、重点与分层随机检查 | `ai_agents.py`、`ai_screening.py` |
| 恢复、证据隔离、停止和状态检查 | `ai_recovery.py`、`ai_resilience.py`、`ai_quarantine.py`、`ai_stops.py`、`ai_check.py`、`progress.py` |
| 统计、图表和执行审计 | `analysis.py`、`structure.py`、`ai_analysis.py`、`ai_screening_report.py`、`ai_figures.py`、`ai_audit.py`、`ai_audit_figures.py`、`ai_outcomes.py` |

仅复现已有结果不必执行下面的步骤。开展新实验时，复制 `.env.example` 为 `.env`，填写自己的 `YOUTUBE_API_KEY` 和模型服务的 `AI_API_KEY`、`AI_BASE_URL`、`AI_MODEL`；在 `config.toml` 设置新的 `run_id`，并按服务调整输出格式、token 参数、并发和速率。

```bash
cp .env.example .env
# 编辑 .env 和 config.toml 后，采集与清洗：
uv run --locked python run.py collect-all
uv run --locked python run.py clean
uv run --locked python run.py analyze-structure

# 准备编码输入、导出提示词和检查配置（这些命令不请求模型）：
uv run --locked python run.py agent-prepare --experiment screening_v1
uv run --locked python run.py agent-plan --experiment screening_v1
uv run --locked python run.py agent-prompts --experiment screening_v1
uv run --locked python run.py agent-check --experiment screening_v1

# 真实模型调用，先使用有限请求预算：
uv run --locked python run.py agent-run --experiment screening_v1 --max-requests 10 --workers 2
uv run --locked python run.py agent-status --experiment screening_v1
# 按需要增加后续请求预算，同一实验可继续运行：
uv run --locked python run.py agent-run --experiment screening_v1 --max-requests 500 --workers 16
# 编码完成后重建统计报告：
uv run --locked python run.py agent-analyze --experiment screening_v1
```

`collect-all` 获取配置年份的可见视频与评论，同一命令支持续传。`agent-prepare` 默认使用全部受众线程；可用 `--transcripts path/to/transcripts.csv` 提供实际已有字幕，列为 `video_id,transcript,transcript_source`。

默认 `workflow = "screening"`：全量初编后检查重点条目，并对非重点部分分层随机抽查 10%；不是全量盲复核或三轮裁决。检查模型不查看初编答案。程序校验 ID、标签结构和逐字证据，分歧及证据不足按协议保留未决或缺失。`--max-requests` 包括重试；`0` 为不限，研究配置的默认预算为不限，运行时建议显式设置预算。

中断后重跑相同实验命令复用合格缓存；修改模型、提示词、输入或研究口径时使用新实验名。原始 SQLite、请求账本、凭证、运行日志和论文材料不在本仓库中，当前 CSV 不能直接用于原实验的数据库续传。新采集时点和模型响应可能不同，不保证得到相同输入与标签。完整命令选项可执行 `uv run --locked python run.py --help` 查看。

## 方法边界与许可

数据描述采集时可见内容及当时累计计数。平台评论计数与实际抓取条数口径不同；关闭、删除、审核或未采完不能视为零评论。AI 检查一致性不等于人工准确率，统计区间不传播 AI 编码误差，观察性关联不证明认知改变、真实分享或因果机制。

原创代码与本说明采用 [MIT](LICENSE)。第三方平台内容和依赖保留各自权利，数据不因代码许可证而获得额外授权。本项目为独立研究工具，不代表 CGTN 或 YouTube 官方立场。
