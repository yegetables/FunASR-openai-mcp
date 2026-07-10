# FunASR OpenAI MCP 扩展

基于 [FunASR](https://github.com/modelscope/FunASR) 的扩展版本，增加 OpenAI 兼容 API 和 MCP 集成。

## 与官方版本的区别

### OpenAI API Server (`examples/openai_api/server.py`)

| 修改 | 说明 |
|------|------|
| +`qwen3-asr` 模型 | MCP 支持 52 语种最高精度模型 |
| +`spk` 参数 | API 支持说话人分离（cam++） |
| `--model` 默认 `None` | 按需加载模型，节省显存 |

官方版本保留为 `server_official.py`。

### MCP Server (`examples/mcp_server/funasr_mcp.py`)

| 修改 | 说明 |
|------|------|
| HTTP 代理模式 | 不加载本地模型，转发到 OpenAI API 服务 |
| +`model` 参数 | 支持所有 5 个模型 |
| +`spk` 参数 | 支持说话人分离 |
| 自动选模型 | 根据语言自动选择最佳模型 |

官方版本保留为 `funasr_mcp_official.py`。

### Streaming (`examples/voice_input/funasr_streaming.py`)

新增实时麦克风/系统音频流式转写，基于 Paraformer-Streaming。

## 使用方式

### 1. 启动 OpenAI API 服务

```bash
# 按需加载模型（推荐）
python examples/openai_api/server.py --device cuda --port 8221

# 预加载指定模型
python examples/openai_api/server.py --device cuda --model sensevoice --port 8221
```

### 2. 配置 Hermes MCP

```bash
hermes mcp add funasr --command python --args "examples/mcp_server/funasr_mcp.py"
```

### 3. 实时麦克风转写

```bash
# 麦克风模式
python examples/voice_input/funasr_streaming.py

# 系统音频模式（直播/视频）
python examples/voice_input/funasr_streaming.py --mode system
```

## 模型列表

| 模型 | 语言 | 速度 | 特点 |
|------|------|------|------|
| sensevoice | 5 种 | 极快 | 情感标签，默认 |
| paraformer | 中/英 | 快 | 中文生产级 |
| paraformer-en | 英 | 快 | 英文专用 |
| fun-asr-nano | 31 种 | 中等 | LLM-based |
| qwen3-asr | 52 种 | 较慢 | 最高精度 |

## 同步上游

```bash
git fetch upstream
git merge upstream/main
```
