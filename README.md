# FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rates

[![arXiv Paper](https://img.shields.io/badge/arXiv_Paper-2606.31247-b31b1b)](https://arxiv.org/abs/2606.31247)
[![demo page](https://img.shields.io/badge/Demo_Page-Github.io-blue)](https://flexislm.github.io)
[![dataset](https://img.shields.io/badge/Data-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/FlexiSLM/datasets)
[![model](https://img.shields.io/badge/Models-green?logo=huggingface&logoColor=white)](https://huggingface.co/FlexiSLM/models)
[![WeChat Blog](https://img.shields.io/badge/WeChat-Blog-07C160?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/pno08CK1dXinIfbvt-v5dg)
[![Presentation Video](https://img.shields.io/badge/Presentation-Video-8A2BE2)](https://jiaqili3.github.io/assets/videos/showcase_video.mp4)

## Overview

This repository contains the code for our paper, "FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rates," along with instructions for downloading the released training data.

FlexiSLM is the first spoken language model that supports *dynamic* and *controllable* frame rates on both speech input and output. A single trained model can be steered between 12.5 Hz and 4.0 Hz without retraining, while its dynamic frame-rate mechanism adapts to the varying complexity of speech. FlexiSLM matches state-of-the-art 7B models even in reduced 6.25Hz frame rates. It also supports controllable frame rate generation.


<!-- ![FlexiSLM architecture](assets/flexislm_architecture.png) -->

## News

- **August 21, 2026:** FlexiSLM is accepted to EMNLP 2026 Main Conference!
- **August 20, 2026: Checkpoint release.** We released the [FlexiSLM-7B Stage 2](https://huggingface.co/FlexiSLM/FlexiSLM-7B-Stage2) checkpoint and [FlexiSLM-0.5B Stage 2](https://huggingface.co/FlexiSLM/FlexiSLM-0_5B-Stage2) checkpoint reproduced with this codebase. Please note that this project is in active development. We expect these checkpoints will be overwritten in the coming days as we train for more steps.
- **August 6, 2026: Data release.** We released [FlexiSLM-Data-4M-s2s](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s), [FlexiSLM-Data-2M-s2s-compact](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact), and [FlexiSLM-Data-5M-t2t](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-5M-t2t).
- **August 2, 2026: Code release.** We released the FlexiSLM-7B training and inference code.

## Installation

```bash
git clone --recurse-submodules https://github.com/AmphionTeam/FlexiSLM.git
cd FlexiSLM
pip install -r requirements.txt
```
## Table of Contents

- [FlexiSLM-Data Details](#flexislm-data-details)
- [Inference Guide](#inference)
- [Training Guide](#training-guide)
- [Evaluation with Kimi-Audio-Evalkit](#evaluation-with-kimi-audio-evalkit)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Appendix: Project File Structure](#appendix-project-structure)

## FlexiSLM-Data Details

We open-source the data produced by the following pipeline:

1. **Prompt collection and response generation.** Text prompts are collected from public QA, instruction-following, and dialogue datasets. Responses are generated with Qwen3-Omni-30B-A3B. The resulting text pairs are released as [![dataset](https://img.shields.io/badge/Data-5M_text2text-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-5M-t2t).
2. **Speech synthesis.** Responses are synthesized with Qwen3-TTS, while prompts are synthesized with Fish-Audio using randomly sampled speaker prompts. The resulting 4.2M samples and approximately 26K hours of audio are released as [![dataset](https://img.shields.io/badge/Data-4M_speech2speech-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s). Download size is about 2.8TB.
3. **Quality filtering and compression.** Stricter filtering is applied and all audio is converted to MP3. The compact release contains 2.43M samples and approximately 14.8K hours of audio in about 385 GB: [![dataset](https://img.shields.io/badge/Data-2M_speech2speech-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact) **We use this compact dataset for training**.

We believe this is one of the largest open-source datasets for spoken language model training, and we hope this will especially benefit new researchers in this area. For data preview and statistics, please refer to the links above.


## Inference

Use the `checkpoint` flag to select an inference checkpoint. The default is **`stage2_7B`**.

| Flag | Hugging Face repo | Manual local directory |
| --- | --- | --- |
| `stage2_7B` (default) | [FlexiSLM/FlexiSLM-7B-Stage2](https://huggingface.co/FlexiSLM/FlexiSLM-7B-Stage2) | `models/FlexiSLM-7B-Stage2` |
| `stage2_0.5B` | [FlexiSLM/FlexiSLM-0_5B-Stage2](https://huggingface.co/FlexiSLM/FlexiSLM-0_5B-Stage2) | `models/FlexiSLM-0_5B-Stage2` |

### 1. Python API (with Automatic downloading)

Set `auto_download=True` to download the selected Stage 2 checkpoint (`stage2_7B` by default, or `stage2_0.5B`), plus the Qwen2.5-Omni audio encoder, SenseVoice, FlexiCodec, flow-matching decoder, and vocoder files into `models/` on first run. Later runs reuse the local copies. 

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
    input_framerate=8.0,
    default_framerate=8.0,
    decode_audio=True,
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
    # Flow-matching Vocos is 24 kHz; FlexiCodec AR decode is 16 kHz.
    sample_rate = int(result.get("sample_rate") or 24_000)
    sf.write(Path(output_path), waveform.squeeze(), sample_rate)


# Text-to-speech
result = engine.generate_tts(
    sentence="FlexiSLM supports controllable speech generation.",
    framerate=8.0,
)
save_audio(result, "tts.wav")

# Automatic speech recognition
result = engine.generate_from_audio(
    audio_path="examples/input.wav",
    text_query="Please transcribe the audio.",
    framerate=8.0,
    output_text_only=True,
)
print(result["text"])

# Audio question answering
result = engine.generate_from_audio(
    audio_path="examples/question.wav",
    text_query="",
    framerate=8.0,
    output_text_only=True,
)
print(result["text"])

# Speech-to-speech generation
result = engine.generate_from_audio(
    audio_path="examples/input.wav",
    text_query="",
    framerate=8.0,
    output_text_only=False,
)
save_audio(result, "s2s.wav")
```

### 2. Python API (Manual downloading)

Download the checkpoint you want to run. Auxiliary encoder and codec files are shared by both sizes.

```bash
MODEL_ROOT="$PWD/models"

# if you want to run stage2_7B
hf download FlexiSLM/FlexiSLM-7B-Stage2 --local-dir "$MODEL_ROOT/FlexiSLM-7B-Stage2"
# if you want to run stage2_0.5B
hf download FlexiSLM/FlexiSLM-0_5B-Stage2 --local-dir "$MODEL_ROOT/FlexiSLM-0_5B-Stage2"

# these files are shared by both sizes
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

Then reuse the Python API example from [Section 1](#1-python-api-with-automatic-downloading), replacing only the `config = FlexiSLMInferenceConfig(...)` block. Set `checkpoint` to match the weights you downloaded, or pass `model_path` directly:

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
    input_framerate=8.0,
    default_framerate=8.0,
    decode_audio=True,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
```

A minimal notebook is available at `examples/inference.ipynb`. 

### 3. Batch Inference

Batch inference reads requests from JSONL and uses a YAML file for model, input, output, and multi-GPU runtime settings. Committed examples are `examples/requests.jsonl` and `examples/infer_7b.yaml`.

`examples/requests.jsonl`:

```jsonl
{"index": 0, "task": "tts", "input": {"text": "FlexiSLM supports controllable speech generation."}, "metadata": {"sample_id": "tts-demo"}}
{"index": 1, "task": "asr", "input": {"audio_path": "examples/input.wav"}, "metadata": {"sample_id": "asr-demo"}}
{"index": 2, "task": "audio_qa", "input": {"audio_path": "examples/question.wav", "model_prompt": ""}, "metadata": {"sample_id": "qa-demo"}}
{"index": 3, "task": "s2s", "input": {"audio_path": "examples/input.wav"}, "metadata": {"sample_id": "s2s-demo"}}
```

`examples/infer_7b.yaml` (abbreviated; see the file for the full config):

```yaml
engine:
  config:
    checkpoint: stage2_7B  # or stage2_0.5B
    model_path: models/FlexiSLM-7B-Stage2  # or models/FlexiSLM-0_5B-Stage2
    qwen25o_encoder_path: models/Qwen2_5-Omni-Audio_Encoder
    # ... encoder / FlexiCodec / SenseVoice / flow-matching paths ...
    use_flow_matching_decoder: true
    enable_flexible_framerate: true
    input_framerate: 8.0
    default_framerate: 8.0
    decode_audio: true
    output_sample_rate: 24000  # Vocos native; use 16000 only for FlexiCodec AR
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

Run after downloading the Stage 2 checkpoint and shared encoder/codec files (see [Section 2](#2-python-api-manual-downloading)):

```bash
python -m src.infer examples/infer_7b.yaml
```

`input`/`output` paths are resolved relative to the repository root. `engine.config` model paths and JSONL `audio_path` values are relative to the working directory (run from the repo root). `engine.config.checkpoint` selects `stage2_7B` or `stage2_0.5B`. `inference.checkpoint` is the local weights path recorded in traces. To fetch weights automatically instead of setting `model_path`, use `engine.config.auto_download: true` with `engine.config.checkpoint: stage2_7B` or `stage2_0.5B`. Optional `inference.transcribe_model_path` (for example `models/whisper-large-v3`) ASR-transcribes generated s2s audio; download Whisper first if you enable it.

The runner writes one unified JSONL trace and stores generated speech under `output.audio_dir`.


## Training Guide

FlexiSLM training has three stages:

1. **Talker and input-module pre-training.** Freeze the Qwen backbone and train the Talker, audio embeddings, and input frame-merging module.
2. **Multi-task LoRA fine-tuning.** Train the Talker and input modules while adapting the Thinker with LoRA.
3. **Full fine-tuning.** Merge the Stage 2 LoRA weights into the Thinker, enable the Talker-to-Thinker connection, and train all model components.

Our released checkpoints are trained with the same settings using 8 A100 GPUs. The configs here are also adapted for 8 A100 GPUs.

### 1. Download additional checkpoints and dataset
```bash
MODEL_ROOT="$PWD/models"
TRAIN_DATA_ROOT="$PWD/data/training"
BENCHMARK_DATA_ROOT="$PWD/data/benchmarks"

# Previously downloaded in inference guide
hf download FlexiSLM/FlexiSLM-7B-Stage2 --local-dir "$MODEL_ROOT/FlexiSLM-7B-Stage2"
hf download FlexiSLM/FlexiSLM-0_5B-Stage2 --local-dir "$MODEL_ROOT/FlexiSLM-0_5B-Stage2"
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

### 2. Data Configuration

The committed recipes use the datasets downloaded in [Section: Download additional checkpoints and dataset](#1-download-additional-checkpoints-and-dataset):

- `config/datasets/train_stage1.yaml` for Stage 1 (ASR+TTS only)
- `config/datasets/train_stage2_3.yaml` for Stages 2 and 3 (ASR+TTS + S2S + WebQ/Trivia; shard retention ratios TTS 0.5, ASR 0.5, S2S 1.0, WebQ/Trivia 3.0). `data_part2_webq_trivia` is included in the same [FlexiSLM-Data-2M-s2s-compact](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact/tree/main/data_part2_webq_trivia) download.

Each training recipe sets `webdataset_steps_per_epoch` for an 8-GPU launch at that recipe's `per_device_train_batch_size`. Recompute it when changing the GPU count or batch size: `ceil(logical_samples / (num_gpus * per_device_train_batch_size))`.
### 3. Launch Training

Training arguments are stored in YAML files under `config/`; launchers live under `scripts/`:

| Stage | Configuration | Launcher | Initialization |
| --- | --- | --- | --- |
| Stage 1 (7B) | `config/train_stage1_7B.yaml` | `scripts/train_stage1_7B.sh` | Qwen2.5-7B base model |
| Stage 2 (7B) | `config/train_stage2_7B.yaml` | `scripts/train_stage2_7B.sh` | released Stage 1 ([Hub](https://huggingface.co/FlexiSLM/FlexiSLM-7B-Stage1)) |
| Stage 3 (7B) | `config/train_stage3_7B.yaml` | `scripts/train_stage3_7B.sh` | merged Stage 2 checkpoint |
| Stage 1 (0.5B) | `config/train_stage1_0_5B.yaml` | `scripts/train_stage1_0_5B.sh` | Qwen2.5-0.5B base model |
| Stage 2 (0.5B) | `config/train_stage2_0_5B.yaml` | `scripts/train_stage2_0_5B.sh` | released Stage 1 ([Hub](https://huggingface.co/FlexiSLM/FlexiSLM-0_5B-Stage1)) |
| Stage 3 (0.5B) | `config/train_stage3_0_5B.yaml` | `scripts/train_stage3_0_5B.sh` | merged 0.5B Stage 2 checkpoint |

Stage 2 sets `resume_from_checkpoint` to the released Stage 1 Hub repo (downloaded into `models/` if missing). Launch each stage after updating its YAML:

```bash
bash scripts/train_stage1_7B.sh
bash scripts/train_stage2_7B.sh
bash scripts/train_stage3_7B.sh
```

The 0.5B recipes use `Qwen2.5-0.5B-Instruct` and larger per-device batches:

```bash
bash scripts/train_stage1_0_5B.sh
bash scripts/train_stage2_0_5B.sh
bash scripts/train_stage3_0_5B.sh
```

YAML values can be overridden on the command line:

```bash
bash scripts/train_stage2_7B.sh \
  --resume_from_checkpoint FlexiSLM/FlexiSLM-7B-Stage1 \
  --output_dir outputs/train_stage2_7B \
  --learning_rate 2e-5
```

Use the shared launcher for a custom configuration:

```bash
bash scripts/train.sh config/train_stage2_7B.yaml
```

The shared launcher uses Accelerate by default. Stage 3 selects DeepSpeed with `config/ds_config_zero2.json`; set `FLEXISLM_LAUNCHER` and `DEEPSPEED_CONFIG` to override the launcher or ZeRO configuration. GPU and distributed settings are detected by `scripts/env.sh`.

## Evaluation with Kimi-Audio-Evalkit

FlexiSLM uses the bundled [Kimi-Audio-Evalkit](https://github.com/petrichor20211/Kimi-Audio-Evalkit) submodule to evaluate VoiceBench, OpenAudioBench, and LibriSpeech. Inference and scoring are separate. Run all commands below from the FlexiSLM repository root.

**Use a separate Python environment for Evalkit scoring.** The Evalkit dependency set (see `Kimi-Audio-Evalkit/requirements.txt`) pins packages such as `sacrebleu==1.5.1` and an older PyTorch stack that conflict with FlexiSLM training/inference. Keep your training/inference env (for example `pip install -r requirements.txt`) unchanged, and install Evalkit requirements only in a dedicated env such as `kimi-audio-evalkit` or `eval`.

A fresh `--recurse-submodules` clone already contains the Evalkit. For an existing clone, initialize it with `git submodule update --init --recursive`. Then create/activate the Evalkit env, install its requirements, and download the benchmark data:

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

### 1. Build Requests and Run Inference

Build requests and run FlexiSLM inference in your **training/inference** environment (not the Evalkit env):

```bash
# activate the FlexiSLM train/infer env first
python local/build_vb_oab_requests.py \
  --benchmark voicebench \
  --data-root data/benchmarks \
  --out-dir outputs/evaluation/requests/voicebench \
  --task audio_qa
python local/build_vb_oab_requests.py \
  --benchmark openaudiobench \
  --data-root data/benchmarks \
  --out-dir outputs/evaluation/requests/openaudiobench \
  --task audio_qa
python local/build_librispeech_requests.py \
  --data-root data/benchmarks \
  --out-dir outputs/evaluation/requests/librispeech
```

Run all ten inference jobs with the committed configuration. The launcher detects the available GPUs and writes every trace to the path expected by `config/eval_benchmarks.yaml`:

```bash
bash scripts/infer_benchmarks.sh
```

### 2. Scoring

Switch to the **Evalkit** environment, export the DeepSeek API key, and run the committed evaluation configuration:

```bash
conda activate kimi-audio-evalkit
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export PYTHONPATH="$PWD:$PWD/Kimi-Audio-Evalkit${PYTHONPATH:+:$PYTHONPATH}"
python -m src.eval config/eval_benchmarks.yaml
```

Results are written to `outputs/evaluation/results/`. To run selected jobs only, repeat `--job`, for example:

```bash
python -m src.eval config/eval_benchmarks.yaml \
  --job voicebench_openbookqa \
  --job librispeech_test-clean
```

The committed configuration covers VoiceBench, OpenAudioBench, and LibriSpeech `test-clean`/`test-other` ASR WER. Use `python -m src.eval --help` for all job selection options. LibriSpeech WER scoring does not need `DEEPSEEK_API_KEY`; VoiceBench / OpenAudioBench judge jobs do.

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{li2026flexislmdynamiccontrollableframe,
      title={FlexiSLM: A Dynamic and Controllable Frame Rate Spoken Language Model},
      author={Jiaqi Li and Chaoren Wang and Xiaohai Tian and Mingjie Chen and Xinyu Liang and Xu Li and Yufan Lin and Junwen Qiu and Jun Zhang and Lu Lu and Haizhou Li and Zhizheng Wu},
      year={2026},
      eprint={2606.31247},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2606.31247},
}
```

## Acknowledgements

- Our work uses Qwen 2.5 as the backbone and [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) as the audio encoder.
- Our training framework is largely based on [Transformers](https://github.com/huggingface/transformers).
- Our evaluation uses [Kimi-Audio-Evalkit](https://github.com/MoonshotAI/Kimi-Audio-Evalkit)
- Our previous open-source works [FlexiCodec](https://github.com/AmphionTeam/FlexiCodec) and [DualCodec](https://github.com/jiaqili3/DualCodec) are foundational to this work.

## License

This project is licensed under the MIT License.

## Appendix: Project Structure
If you want to understand the project structure, you can refer to the following:
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

