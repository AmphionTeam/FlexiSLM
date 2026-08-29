# FlexiSLM: 支持动态与可控帧率的口语语言模型

[English](README.md) | **中文**

[![arXiv Paper](https://img.shields.io/badge/arXiv_Paper-2606.31247-b31b1b)](https://arxiv.org/abs/2606.31247)
[![demo page](https://img.shields.io/badge/Demo_Page-Github.io-blue)](https://flexislm.github.io)
[![dataset](https://img.shields.io/badge/Data-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/FlexiSLM/datasets)
[![model](https://img.shields.io/badge/Models-green?logo=huggingface&logoColor=white)](https://huggingface.co/FlexiSLM/models)
[![WeChat Blog](https://img.shields.io/badge/WeChat-Blog-07C160?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/pno08CK1dXinIfbvt-v5dg)

## 概述

本仓库包含论文 *FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rates* 的代码，以及已发布训练数据的下载说明。

FlexiSLM 是首个在语音输入与输出两端均支持**动态**且**可控**帧率的口语语言模型。同一套训练好的模型可在无需重新训练的情况下，在 12.5 Hz 与 4.0 Hz 之间调节；其动态帧率机制会随语音复杂度变化而自适应调整。即便将帧率降至 6.25 Hz，FlexiSLM 仍能与当前最先进的 7B 模型持平，并支持可控帧率生成。


<!-- ![FlexiSLM architecture](assets/flexislm_architecture.png) -->

## 新闻

- **2026 年 8 月 21 日：** FlexiSLM 被 EMNLP 2026 主会接收！
- **2026 年 8 月 20 日：检查点发布。** 我们发布了基于本代码库复现的 [FlexiSLM-7B Stage 2](https://huggingface.co/FlexiSLM/FlexiSLM-7B-Stage2) 检查点与 [FlexiSLM-0.5B Stage 2](https://huggingface.co/FlexiSLM/FlexiSLM-0_5B-Stage2) 检查点。
- **2026 年 8 月 6 日：数据发布。** 我们发布了 [FlexiSLM-Data-4M-s2s](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s)、[FlexiSLM-Data-2M-s2s-compact](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact) 与 [FlexiSLM-Data-5M-t2t](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-5M-t2t)。
- **2026 年 8 月 2 日：代码发布。** 我们发布了 FlexiSLM-7B 的训练与推理代码。

## 安装

```bash
git clone --recurse-submodules https://github.com/AmphionTeam/FlexiSLM.git
cd FlexiSLM
pip install -r requirements.txt
```
## 目录

- [FlexiSLM-Data 数据详情](#flexislm-data-数据详情)
- [推理指南](#推理)
- [训练指南](#训练指南)
- [使用 Kimi-Audio-Evalkit 评测](#使用-kimi-audio-evalkit-评测)
- [已发布检查点的评测结果](#评测结果)
- [引用](#引用)
- [致谢](#致谢)
- [项目文件结构](#项目结构)

## FlexiSLM-Data 数据详情

我们开源了由以下流程产出的数据：

1. **Prompt 收集与回复生成。** 文本 prompt 来自公开的问答、指令跟随与对话数据集。回复由 Qwen3-Omni-30B-A3B 生成。得到的文本对发布为 [![dataset](https://img.shields.io/badge/Data-5M_text2text-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-5M-t2t)。
2. **语音合成。** 回复使用 Qwen3-TTS 合成，prompt 使用 Fish-Audio 并随机采样说话人 prompt 合成。得到的 420 万条样本、约 2.6 万小时音频发布为 [![dataset](https://img.shields.io/badge/Data-4M_speech2speech-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s)。下载体积约 2.8TB。
3. **质量过滤与压缩。** 采用更严格的过滤，并将全部音频转为 MP3。精简版包含 243 万条样本、约 1.48 万小时音频，体积约 385 GB：[![dataset](https://img.shields.io/badge/Data-2M_speech2speech-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact) **训练使用该精简数据集**。

我们相信这是目前规模最大的口语语言模型开源训练数据之一，希望尤其能帮助该领域的新研究者。数据预览与统计请见上方链接。


## 推理

使用 `checkpoint` 参数选择推理检查点。默认值为 **`stage2_7B`**。

| 参数 | Hugging Face 仓库 | 下载目录 |
| --- | --- | --- |
| `stage2_7B`（默认） | [FlexiSLM/FlexiSLM-7B-Stage2](https://huggingface.co/FlexiSLM/FlexiSLM-7B-Stage2) | `models/FlexiSLM-7B-Stage2` |
| `stage2_0.5B` | [FlexiSLM/FlexiSLM-0_5B-Stage2](https://huggingface.co/FlexiSLM/FlexiSLM-0_5B-Stage2) | `models/FlexiSLM-0_5B-Stage2` |

### 1. Python API（自动下载）

将 `auto_download=True` 后，首次运行会下载所选 Stage 2 检查点（默认 `stage2_7B`，也可选 `stage2_0.5B`），以及 Qwen2.5-Omni 音频编码器、SenseVoice、FlexiCodec、流匹配解码器与声码器文件到 `models/`。之后运行会复用本地副本。

```python
from pathlib import Path
import soundfile as sf
import torch

from src.inference_flexislm import (
    FlexiSLMInferenceConfig,
    FlexiSLMInference,
)

config = FlexiSLMInferenceConfig(
    auto_download=True,
    checkpoint="stage2_7B",  # or "stage2_0.5B"
    use_flow_matching_decoder=True,
    flow_matching_prompt_audio_path=str(
        Path("examples/input.wav").resolve()
    ),
    enable_flexible_framerate=True,
    default_input_framerate=8.0,
    default_output_framerate=8.0,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
engine = FlexiSLMInference(config, device="cuda:0")


def save_audio(result, output_path):
    waveform = result.get("audio")
    if waveform is None:
        raise RuntimeError("The model did not return decoded audio")
    if torch.is_tensor(waveform):
        waveform = waveform.detach().float().cpu().numpy()
    sf.write(Path(output_path), waveform.squeeze(), 24_000)


# Text-to-speech
result = engine.generate_tts(
    sentence="This is a test sentence.",
    output_framerate=8.0,
)
save_audio(result, "tts.wav")

# Automatic speech recognition
result = engine.generate_from_audio(
    audio_path="examples/input.wav",
    text_query="Please transcribe the audio.",
    input_framerate=8.0,
    output_framerate=8.0,
    output_text_only=True,
)
print(result["text"])

# Audio question answering
result = engine.generate_from_audio(
    audio_path="examples/question.wav",
    text_query="",
    input_framerate=8.0,
    output_framerate=8.0,
    output_text_only=True,
)
print(result["text"])

# Speech-to-speech generation
result = engine.generate_from_audio(
    audio_path="examples/input.wav",
    text_query="",
    input_framerate=8.0,
    output_framerate=8.0,
    output_text_only=False,
)
save_audio(result, "s2s.wav")
```

### 2. Python API（手动下载）

下载要运行的检查点。辅助编码器与 codec 文件两种规模共用。

```bash
MODEL_ROOT="$PWD/models"

# if you want to run stage2_7B
hf download FlexiSLM/FlexiSLM-7B-Stage2 --local-dir "$MODEL_ROOT/FlexiSLM-7B-Stage2"
# if you want to run stage2_0.5B
hf download FlexiSLM/FlexiSLM-0_5B-Stage2 --local-dir "$MODEL_ROOT/FlexiSLM-0_5B-Stage2"

# required files
hf download FlexiSLM/Qwen2_5-Omni-Audio_Encoder --local-dir "$MODEL_ROOT/Qwen2_5-Omni-Audio_Encoder"
hf download FunAudioLLM/SenseVoiceSmall --local-dir "$MODEL_ROOT/SenseVoiceSmall"
hf download jiaqili3/flexicodec \
  12hz_v1_half_config.yaml \
  nartts_flexicodec_only.safetensors \
  nartts.safetensors \
  --local-dir "$MODEL_ROOT/FlexiCodec"
hf download amphion/dualcodec-tts vocos_emilia.safetensors \
  --local-dir "$MODEL_ROOT/FlexiCodec"
```

然后复用 [第 1 节](#1-python-api自动下载) 中的 Python API 示例，只需替换 `config = FlexiSLMInferenceConfig(...)` 代码块。将 `checkpoint` 设为与已下载权重一致，或直接传入 `model_path`：

```python
model_root = Path.cwd() / "models"
config = FlexiSLMInferenceConfig(
    checkpoint="stage2_7B",  # or "stage2_0.5B" → models/FlexiSLM-0_5B-Stage2
    model_path=str(model_root / "FlexiSLM-7B-Stage2"),  # or model_root / "FlexiSLM-0_5B-Stage2"
    qwen25o_encoder_path=str(model_root / "Qwen2_5-Omni-Audio_Encoder"),
    qwen25o_encoder_config_path=str(
        model_root / "Qwen2_5-Omni-Audio_Encoder/config.json"
    ),
    flexicodec_ckpt_path=str(
        model_root / "FlexiCodec/nartts_flexicodec_only.safetensors"
    ),
    flexicodec_config_path=str(model_root / "FlexiCodec/12hz_v1_half_config.yaml"),
    sensevoice_path=str(model_root / "SenseVoiceSmall"),
    use_flow_matching_decoder=True,
    flow_matching_ckpt_path=str(model_root / "FlexiCodec/nartts.safetensors"),
    flow_matching_vocoder_path=str(
        model_root / "FlexiCodec/vocos_emilia.safetensors"
    ),
    flow_matching_prompt_audio_path=str(
        Path("examples/input.wav").resolve()
    ),
    enable_flexible_framerate=True,
    default_input_framerate=8.0,
    default_output_framerate=8.0,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
```

精简 notebook 见 `examples/inference.ipynb`。

### 3. 批量推理

批量推理从 JSONL 读取请求，并用 YAML 配置模型、输入、输出与多 GPU 运行设置。仓库中的示例为 `examples/requests.jsonl` 与 `examples/infer_7b.yaml`。

`examples/requests.jsonl`：

```jsonl
{"index": 0, "task": "tts", "input": {"text": "FlexiSLM supports controllable speech generation."}, "metadata": {"sample_id": "tts-demo"}}
{"index": 1, "task": "asr", "input": {"audio_path": "examples/input.wav"}, "metadata": {"sample_id": "asr-demo"}}
{"index": 2, "task": "audio_qa", "input": {"audio_path": "examples/question.wav", "model_prompt": ""}, "metadata": {"sample_id": "qa-demo"}}
{"index": 3, "task": "s2s", "input": {"audio_path": "examples/input.wav"}, "metadata": {"sample_id": "s2s-demo"}}
```

`examples/infer_7b.yaml`（节选；完整配置见该文件）：

```yaml
engine:
  config:
    checkpoint: stage2_7B  # or stage2_0.5B
    model_path: models/FlexiSLM-7B-Stage2  # or models/FlexiSLM-0_5B-Stage2
    qwen25o_encoder_path: models/Qwen2_5-Omni-Audio_Encoder
    # ... encoder / FlexiCodec / SenseVoice / flow-matching paths ...
    use_flow_matching_decoder: true
    enable_flexible_framerate: true
    default_input_framerate: 8.0
    default_output_framerate: 8.0
    output_sample_rate: 24000
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2

# input/output paths are resolved relative to the repository root
input:
  path: examples/requests.jsonl

output:
  trace_path: outputs/inference/traces.jsonl
  audio_dir: outputs/inference/audio
  error_path: outputs/inference/errors.jsonl

inference:
  checkpoint: models/FlexiSLM-7B-Stage2  # or models/FlexiSLM-0_5B-Stage2
  target_framerate_hz: 8.0
  output_sample_rate: 24000

runtime:
  devices: [cuda:0]
  workers_per_device: 1
  fail_fast: false
```

下载 Stage 2 检查点及共用的编码器 / codec 文件后运行（见 [第 2 节](#2-python-api手动下载)）：

```bash
python -m src.infer examples/infer_7b.yaml
```

`input` / `output` 路径相对于仓库根目录解析。`engine.config` 中的模型路径与 JSONL 中的 `audio_path` 相对于工作目录（请在仓库根目录运行）。`engine.config.checkpoint` 用于选择 `stage2_7B` 或 `stage2_0.5B`。`inference.checkpoint` 是写入 traces 的本地权重路径。若希望自动拉取权重而不设置 `model_path`，可使用 `engine.config.auto_download: true`，并将 `engine.config.checkpoint` 设为 `stage2_7B` 或 `stage2_0.5B`。可选的 `inference.transcribe_model_path`（例如 `models/whisper-large-v3`）会对生成的 s2s 音频做 ASR 转写；若启用，请先下载 Whisper。

运行器会写入一份统一的 JSONL trace，并将生成的语音存到 `output.audio_dir`。


## 训练指南

FlexiSLM 训练分为三个阶段：

1. **Talker 与输入模块预训练。** 冻结 Qwen 骨干网络，训练 Talker、音频嵌入与输入帧合并模块。为获得更好效果，我们发布的检查点曾单独用 TTS 任务训练 Talker 模块，再将其权重合并进 Stage 1 权重。但我们也确认：即便 Stage 1 不含 Talker 权重，Stage 2 的效果仍然很好。
2. **多任务 LoRA 微调。** 训练 Talker 与输入模块，同时用 LoRA 适配 Thinker。
3. **全量微调。** 将 Stage 2 的 LoRA 权重合并进 Thinker，启用 Talker 到 Thinker 的连接，并训练全部模型组件。

我们发布的检查点使用相同设置、在 8 张 A100 GPU 上训练。此处配置同样按 8 张 A100 适配。

### 1. 下载额外检查点与数据集
```bash
MODEL_ROOT="$PWD/models"
TRAIN_DATA_ROOT="$PWD/data/training"
BENCHMARK_DATA_ROOT="$PWD/data/benchmarks"

# Previously downloaded in inference guide
hf download FlexiSLM/Qwen2_5-Omni-Audio_Encoder --local-dir "$MODEL_ROOT/Qwen2_5-Omni-Audio_Encoder"
hf download FunAudioLLM/SenseVoiceSmall --local-dir "$MODEL_ROOT/SenseVoiceSmall"
hf download jiaqili3/flexicodec 12hz_v1_half_config.yaml nartts_flexicodec_only.safetensors --local-dir "$MODEL_ROOT/FlexiCodec"

# Required for training
hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$MODEL_ROOT/Qwen2.5-7B-Instruct"
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$MODEL_ROOT/Qwen2.5-0.5B-Instruct"
hf download openai/whisper-large-v3 --local-dir "$MODEL_ROOT/whisper-large-v3"

# S2S Data for Stage 2 and 3 (includes data/ and data_part2_webq_trivia/)
hf download FlexiSLM/FlexiSLM-Data-2M-s2s-compact \
  --repo-type dataset \
  --local-dir "$TRAIN_DATA_ROOT/FlexiSLM-Data-2M-s2s-compact"

# ASR+TTS Data for Stage 1, 2, and 3
hf download FlexiSLM/asrtts_packed_webdataset \
  --repo-type dataset \
  --local-dir "$TRAIN_DATA_ROOT/asrtts_packed_webdataset"

```

### 2. 启动训练

训练配方将 `report_to` 设为 `swanlab`。启动前请在 shell 中导出 SwanLab API key：

```bash
export SWANLAB_API_KEY="your_swanlab_api_key"
```

训练参数存放在 `config/` 下的 YAML 中；启动脚本在 `scripts/` 下：

| 阶段 | 配置 | 数据配置 | 启动脚本 | 初始化 |
| --- | --- | --- | --- | --- |
| Stage 1（7B） | `config/train_stage1_7B.yaml` | `config/datasets/train_stage1.yaml` | `scripts/train_stage1_7B.sh` | Qwen2.5-7B Instruct 模型 |
| Stage 2（7B） | `config/train_stage2_7B.yaml` | `config/datasets/train_stage2_3.yaml` | `scripts/train_stage2_7B.sh` | 已发布的 Stage 1（[Hub](https://huggingface.co/FlexiSLM/FlexiSLM-7B-Stage1)） |
| Stage 3（7B） | `config/train_stage3_7B.yaml` | `config/datasets/train_stage2_3.yaml` | `scripts/train_stage3_7B.sh` | 合并后的 Stage 2 检查点 |
| Stage 1（0.5B） | `config/train_stage1_0_5B.yaml` | `config/datasets/train_stage1.yaml` | `scripts/train_stage1_0_5B.sh` | Qwen2.5-0.5B Instruct 模型 |
| Stage 2（0.5B） | `config/train_stage2_0_5B.yaml` | `config/datasets/train_stage2_3.yaml` | `scripts/train_stage2_0_5B.sh` | 已发布的 Stage 1（[Hub](https://huggingface.co/FlexiSLM/FlexiSLM-0_5B-Stage1)） |
| Stage 3（0.5B） | `config/train_stage3_0_5B.yaml` | `config/datasets/train_stage2_3.yaml` | `scripts/train_stage3_0_5B.sh` | 合并后的 0.5B Stage 2 检查点 |

Stage 2 将 `resume_from_checkpoint` 设为已发布的 Stage 1 Hub 仓库（若不存在会下载到 `models/`）。更新对应 YAML 后即可启动各阶段：

```bash
bash scripts/train_stage1_7B.sh
bash scripts/train_stage2_7B.sh
bash scripts/train_stage3_7B.sh
# for 0.5B
bash scripts/train_stage1_0_5B.sh
bash scripts/train_stage2_0_5B.sh
bash scripts/train_stage3_0_5B.sh
```

YAML 中的值可在命令行覆盖：

```bash
bash scripts/train_stage2_7B.sh \
  --resume_from_checkpoint FlexiSLM/FlexiSLM-7B-Stage1 \
  --output_dir outputs/train_stage2_7B \
  --learning_rate 2e-5
```

## 使用 Kimi-Audio-Evalkit 评测

FlexiSLM 使用捆绑的 [Kimi-Audio-Evalkit](https://github.com/petrichor20211/Kimi-Audio-Evalkit) 子模块评测 VoiceBench、OpenAudioBench 与 LibriSpeech。推理与打分分开进行。以下命令均请在 FlexiSLM 仓库根目录执行。

**请为 Evalkit 打分使用独立的 Python 环境。** Evalkit 依赖（见 `Kimi-Audio-Evalkit/requirements.txt`）会固定如 `sacrebleu==1.5.1` 以及较旧的 PyTorch 栈，与 FlexiSLM 训练 / 推理环境冲突。请保持训练 / 推理环境（例如 `pip install -r requirements.txt`）不变，仅在独立环境（如 `kimi-audio-evalkit` 或 `eval`）中安装 Evalkit 依赖。

使用 `--recurse-submodules` 全新克隆时已包含 Evalkit。若是已有仓库，请用 `git submodule update --init --recursive` 初始化。然后创建 / 激活 Evalkit 环境，安装依赖并下载评测数据：

```bash
# Example: dedicated conda env (do NOT install this into the FlexiSLM train/infer env)
conda create -n kimi-audio-evalkit python=3.10 -y
conda activate kimi-audio-evalkit
pip install -r Kimi-Audio-Evalkit/requirements.txt

BENCHMARK_DATA_ROOT="$PWD/data/benchmarks"
python Kimi-Audio-Evalkit/data/download_benchmark.py \
  --datasets VoiceBench,OpenAudioBench,LibriSpeech \
  --output-dir "$BENCHMARK_DATA_ROOT"
```

### 1. 构建请求并运行推理

在**训练 / 推理**环境中构建请求并运行 FlexiSLM 推理（不要用 Evalkit 环境）。VoiceBench 与 OpenAudioBench 使用 **s2s**（口语回答，24 kHz WAV）。评测省略 MMSU 与 OpenBookQA。LibriSpeech 仍为 ASR（仅文本）：

```bash
# activate the FlexiSLM train/infer env first
export TRANSCRIBE_MODEL_PATH="${TRANSCRIBE_MODEL_PATH:-models/whisper-large-v3}"
python local/build_vb_oab_requests.py \
  --benchmark voicebench \
  --data-root data/benchmarks \
  --out-dir outputs/evaluation/requests/voicebench \
  --task s2s \
  --transcribe-model-path "$TRANSCRIBE_MODEL_PATH" \
  --subsets sd-qa advbench ifeval alpacaeval_full commoneval
python local/build_vb_oab_requests.py \
  --benchmark openaudiobench \
  --data-root data/benchmarks \
  --out-dir outputs/evaluation/requests/openaudiobench \
  --task s2s \
  --transcribe-model-path "$TRANSCRIBE_MODEL_PATH"
python local/build_librispeech_requests.py \
  --data-root data/benchmarks \
  --out-dir outputs/evaluation/requests/librispeech
```

使用对应帧率的 `CONFIG` 运行推理。若设置了 `CUDA_VISIBLE_DEVICES`，启动脚本会使用其中全部 GPU，否则使用所有可见 GPU（`nvidia-smi`），每个 wave 为每张 GPU 分配一个任务。生成的 s2s WAV 存放在每条 trace 旁的 `audio/` 目录。

```bash
# 12.5 Hz in/out → outputs/evaluation/traces/12_5hz/
CONFIG=config/infer_benchmarks_12_5hz.yaml bash scripts/infer_benchmarks.sh

# 6.25 Hz in/out → outputs/evaluation/traces/6_25hz/
CONFIG=config/infer_benchmarks_6_25hz.yaml bash scripts/infer_benchmarks.sh
```

### 2. 打分

切换到 **Evalkit** 环境，导出 DeepSeek API key，并运行与推理帧率匹配的评测 YAML。YAML 已列出全部任务，无需再传 `--job`。

```bash
conda activate kimi-audio-evalkit
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export PYTHONPATH="$PWD:$PWD/Kimi-Audio-Evalkit${PYTHONPATH:+:$PYTHONPATH}"

# After CONFIG=config/infer_benchmarks_12_5hz.yaml
python -m src.eval config/eval_benchmarks_12_5hz.yaml

# After CONFIG=config/infer_benchmarks_6_25hz.yaml
python -m src.eval config/eval_benchmarks_6_25hz.yaml
```

结果写入 `outputs/evaluation/results/{12_5,6_25}hz/`。LibriSpeech WER 不需要 `DEEPSEEK_API_KEY`；VoiceBench / OpenAudioBench 的 LLM-judge 任务需要。

## 评测结果

我们使用 Deepseek-V4-Flash-0731 作为裁判模型，并基于已发布检查点评测。输入与输出帧率设为相同。

下表数字与上文指南中的 DeepSeek 裁判设置一致。对于 FlexiSLM 的 **s2s** traces，**s2t** 是模型直接文本通道（`output.text`），**s2s** 是对生成口语回答做 Whisper ASR 的结果。Qwen2.5-Omni 作为同一裁判下的基线。FlexiSLM-7B Stage 2 分别报告 12.5 Hz 与 6.25 Hz。

| Benchmark | Metric | Qwen2.5-Omni s2t | Qwen2.5-Omni s2s | FlexiSLM-7B-Stage2 12.5 Hz s2t | FlexiSLM-7B-Stage2 12.5 Hz s2s | FlexiSLM-7B-Stage2 6.25 Hz s2t | FlexiSLM-7B-Stage2 6.25 Hz s2s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LibriSpeech | test-clean (WER ↓) | 2.38 | — | 2.14 | — | 3.43 | — |
| | test-other (WER ↓) | 4.21 | — | 5.75 | — | 6.15 | — |
| OpenAudioBench | Llama Questions (Acc ↑) | 76.85 | 72.24 | 80.67 | 74.00 | 80.67 | 72.33 |
| | Web Questions (Acc ↑) | 52.4 | 51.5 | 58.2 | 55.2 | 59.0 | 55.1 |
| | TriviaQA (Acc ↑) | 57.6 | 56.16 | 63.3 | 52.5 | 63.3 | 52.6 |
| VoiceBench | AlpacaEval (Score ↑) | 3.71 | 3.45 | 4.96 | 4.78 | 4.94 | 4.85 |
| | CommonEval (Score ↑) | 3.67 | 3.63 | 4.97 | 4.98 | 4.95 | 4.92 |
| | SD-QA (Acc ↑) | 55.88 | 50.99 | 61.84 | 55.88 | 59.67 | 54.07 |
| | AdvBench (Acc ↑) | - | 98.65 | — | 94.04 | — | 94.42 |


## 引用

如果本工作对你有帮助，欢迎引用：

```bibtex
@article{li2026flexislm,
  title={FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rates},
  author={Li, Jiaqi and Wang, Chaoren and Tian, Xiaohai and Chen, Mingjie and Liang, Xinyu and Li, Xu and Lin, Yufan and Qiu, Junwen and Zhang, Jun and Lu, Lu and others},
  journal={arXiv preprint arXiv:2606.31247},
  year={2026}
}
```

## 致谢

- 本工作以 Qwen 2.5 为骨干网络，并以 [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) 作为音频编码器。
- 训练框架主要基于 [Transformers](https://github.com/huggingface/transformers)。
- 评测使用 [Kimi-Audio-Evalkit](https://github.com/MoonshotAI/Kimi-Audio-Evalkit)
- 我们此前的开源工作 [FlexiCodec](https://github.com/AmphionTeam/FlexiCodec) 与 [DualCodec](https://github.com/jiaqili3/DualCodec) 是本工作的基础。

## 许可证

本项目采用 MIT 许可证。

## 项目结构
若需了解项目结构，可参考如下目录：
```text
FlexiSLM/
├── assets/                 # Documentation and demo assets
├── config/                 # Training and dataset YAML configurations
│   └── datasets/           # Dataset recipes used by training
├── data/                   # Downloaded data
│   ├── benchmarks/         # VoiceBench, OpenAudioBench, and LibriSpeech
│   └── training/           # Released FlexiSLM training datasets
├── examples/               # Inference notebook and small examples
├── local/                  # Data conversion and benchmark request tools
├── Kimi-Audio-Evalkit/     # Evaluation toolkit
├── models/                 # Downloaded models
├── scripts/                # Training launchers and runtime setup
├── src/                    # Model, training, inference, and evaluation code
│   ├── dataset/            # Dataset loading and collation
│   ├── eval/               # Kimi-Audio-Evalkit adapters
│   ├── infer/              # YAML-driven inference runner
│   ├── models/             # FlexiSLM and vendored FlexiCodec implementation
    └── trainer/            # Trainer implementation
```
