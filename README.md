# FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rates

[![arXiv Paper](https://img.shields.io/badge/arXiv_Paper-2606.31247-b31b1b)](https://arxiv.org/abs/2606.31247)
[![demo page](https://img.shields.io/badge/Demo_Page-Github.io-blue)](https://flexislm.github.io)
[![dataset](https://img.shields.io/badge/Data-2M_speech2speech-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact)
[![dataset](https://img.shields.io/badge/Data-4M_speech2speech-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s)

## Table of Contents

- [Inference Guide](#inference)
- [Data Details](#data)
- [Training Guide](#training-guide)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Appendix: Project Structure](#appendix-project-structure)

## Overview

This repository contains the code for our paper, "FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rates."

FlexiSLM is the first spoken language model that supports *dynamic* and *controllable* frame rates on both speech input and output. A single trained model can be steered between 12.5 Hz and 4.0 Hz without retraining, while its dynamic frame-rate mechanism adapts to the varying complexity of speech. FlexiSLM uses a Thinker-Talker architecture with dynamic frame-rate compression on speech input and controllable frame-rate generation on speech output.

<!-- ![FlexiSLM architecture](assets/flexislm_architecture.png) -->

## News

- **August 20, 2026: Checkpoint release.** We released the reproduced [FlexiSLM-7B Stage 2 checkpoint](https://huggingface.co/FlexiSLM/FlexiSLM-7B-Stage2).
- **August 6, 2026: Data release.** We released [FlexiSLM-Data-4M-s2s](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s), [FlexiSLM-Data-2M-s2s-compact](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact), and [FlexiSLM-Data-5M-t2t](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-5M-t2t).
- **August 2, 2026: Code release.** We released the FlexiSLM-7B training and inference code.


## Installation

```bash
git clone --recurse-submodules https://github.com/AmphionTeam/FlexiSLM.git
cd FlexiSLM
pip install -r requirements.txt
```

## Inference

### 1. Python API (with Automatic downloading)

Set `auto_download=True` to download the default FlexiSLM-7B Stage 2 checkpoint, Qwen2.5-Omni audio encoder, SenseVoice, and FlexiCodec files into `models/` on first run. Later runs reuse the local copies.

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
    use_flow_matching_decoder=False,
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
    sf.write(Path(output_path), waveform.squeeze(), 16_000)


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

#### Python API (Manual downloading)

```bash
MODEL_ROOT="$PWD/models"

hf download FlexiSLM/FlexiSLM-7B-Stage2 --local-dir "$MODEL_ROOT/FlexiSLM-7B-Stage2"
hf download FlexiSLM/Qwen2_5-Omni-Audio_Encoder --local-dir "$MODEL_ROOT/Qwen2_5-Omni-Audio_Encoder"
hf download FunAudioLLM/SenseVoiceSmall --local-dir "$MODEL_ROOT/SenseVoiceSmall"
hf download jiaqili3/flexicodec 12hz_v1_half_config.yaml nartts_flexicodec_only.safetensors --local-dir "$MODEL_ROOT/FlexiCodec"
```

Then point the config at those directories:

```python
from pathlib import Path

import soundfile as sf
import torch

from src.inference_flexislm import (
    FlexiSLMInferenceConfig,
    FlexiSLMInference,
)

model_root = Path.cwd() / "models"
config = FlexiSLMInferenceConfig(
    model_path=str(model_root / "FlexiSLM-7B-Stage2"),
    qwen25o_encoder_path=str(model_root / "Qwen2_5-Omni-Audio_Encoder"),
    qwen25o_encoder_config_path=str(
        model_root / "Qwen2_5-Omni-Audio_Encoder/config.json"
    ),
    flexicodec_ckpt_path=str(
        model_root / "FlexiCodec/nartts_flexicodec_only.safetensors"
    ),
    flexicodec_config_path=str(model_root / "FlexiCodec/12hz_v1_half_config.yaml"),
    sensevoice_path=str(model_root / "SenseVoiceSmall"),
    use_flow_matching_decoder=False,
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
    sf.write(Path(output_path), waveform.squeeze(), 16_000)


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

A minimal notebook is available at `examples/inference.ipynb`.

### 2. Batch Inference

Batch inference reads requests from JSONL and uses a YAML file for model, input, output, and multi-GPU runtime settings. Create `examples/requests.jsonl`:

```jsonl
{"index": 0, "task": "tts", "input": {"text": "FlexiSLM supports controllable speech generation."}, "metadata": {"sample_id": "tts-demo"}}
{"index": 1, "task": "asr", "input": {"audio_path": "examples/input.wav"}, "metadata": {"sample_id": "asr-demo"}}
{"index": 2, "task": "audio_qa", "input": {"audio_path": "examples/question.wav", "model_prompt": ""}, "metadata": {"sample_id": "qa-demo"}}
{"index": 3, "task": "s2s", "input": {"audio_path": "examples/input.wav"}, "metadata": {"sample_id": "s2s-demo"}}
```

Create `examples/infer_7b.yaml`:

```yaml
engine:
  config:
    model_path: models/FlexiSLM-7B-Stage2
    qwen25o_encoder_path: models/Qwen2_5-Omni-Audio_Encoder
    qwen25o_encoder_config_path: models/Qwen2_5-Omni-Audio_Encoder/config.json
    flexicodec_ckpt_path: models/FlexiCodec/nartts_flexicodec_only.safetensors
    flexicodec_config_path: models/FlexiCodec/12hz_v1_half_config.yaml
    sensevoice_path: models/SenseVoiceSmall
    use_flow_matching_decoder: false
    enable_flexible_framerate: true
    input_framerate: 8.0
    default_framerate: 8.0
    decode_audio: true
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2

input:
  path: examples/requests.jsonl

output:
  trace_path: outputs/inference/traces.jsonl
  audio_dir: outputs/inference/audio
  error_path: outputs/inference/errors.jsonl

inference:
  checkpoint: models/FlexiSLM-7B-Stage2
  target_framerate_hz: 8.0
  transcribe_model_path: models/whisper-large-v3
  output_sample_rate: 16000

runtime:
  devices: [cuda:0]
  workers_per_device: 1
  fail_fast: false
```

Run the batch inference entrypoint:

```bash
python -m src.infer examples/infer_7b.yaml
```

The runner writes one unified JSONL trace and stores generated speech under `output.audio_dir`.

## Data

We open-source the data produced by the following pipeline:

1. **Prompt collection and response generation.** Text prompts are collected from public QA, instruction-following, and dialogue datasets. Responses are generated with Qwen3-Omni-30B-A3B. The resulting text pairs are released as [FlexiSLM-Data-5M-t2t](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-5M-t2t).
2. **Speech synthesis.** Responses are synthesized with Qwen3-TTS, while prompts are synthesized with Fish-Audio using randomly sampled speaker prompts. The resulting 4.2M samples and approximately 26K hours of audio are released as [FlexiSLM-Data-4M-s2s](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s).
3. **Quality filtering and compression.** Stricter filtering is applied and all audio is converted to MP3. The compact release contains 2.43M samples and approximately 14.8K hours of audio in about 385 GB: [FlexiSLM-Data-2M-s2s-compact](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact).
For more details, please refer to the dataset READMEs on huggingface.

## Training Guide

FlexiSLM training has three stages:

1. **Talker and input-module pre-training.** Freeze the Qwen backbone and train the Talker, audio embeddings, and input frame-merging module.
2. **Multi-task LoRA fine-tuning.** Train the Talker and input modules while adapting the Thinker with LoRA.
3. **Full fine-tuning.** Merge the Stage 2 LoRA weights into the Thinker, enable the Talker-to-Thinker connection, and train all model components.

### Download additional checkpoints and dataset
```bash
MODEL_ROOT="$PWD/models"
TRAIN_DATA_ROOT="$PWD/data/training"
BENCHMARK_DATA_ROOT="$PWD/data/benchmarks"

hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$MODEL_ROOT/Qwen2.5-7B-Instruct"
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$MODEL_ROOT/Qwen2.5-0.5B-Instruct"
hf download openai/whisper-large-v3 --local-dir "$MODEL_ROOT/whisper-large-v3"

hf download FlexiSLM/FlexiSLM-Data-2M-s2s-compact \
  --repo-type dataset \
  --local-dir "$TRAIN_DATA_ROOT/FlexiSLM-Data-2M-s2s-compact"

```

### 1. Data Configuration

The committed recipes use the compact dataset downloaded in [Download additional checkpoints and dataset](#download-additional-checkpoints-and-dataset):

- `config/datasets/train_stage1.yaml` for Stage 1
- `config/datasets/train_stage2_3.yaml` for Stages 2 and 3

Both point to:

```yaml
dataset_backend: webdataset_stream

dataset:
  speech2speech:
    ratio: 1.0
    data_format: webdataset_stream
    data_paths:
      - data/training/FlexiSLM-Data-2M-s2s-compact/data/train-{00000..00242}-of-00243.tar
    webdataset:
      layout: s2s_pair
```

The provided recipe uses 60,645 batches per epoch, corresponding to 2,425,778 training samples with 8 GPUs and 5 samples per GPU. Adjust `webdataset_runtime.sampling.steps_per_epoch` when changing the GPU count or per-device batch size.

### 2. Launch Training

Training arguments are stored in YAML files under `config/`; launchers live under `scripts/`:

| Stage | Configuration | Launcher | Initialization |
| --- | --- | --- | --- |
| Stage 1 (7B) | `config/train_stage1_7B.yaml` | `scripts/train_stage1_7B.sh` | Qwen2.5-7B base model |
| Stage 2 (7B) | `config/train_stage2_7B.yaml` | `scripts/train_stage2_7B.sh` | exported Stage 1 checkpoint |
| Stage 3 (7B) | `config/train_stage3_7B.yaml` | `scripts/train_stage3_7B.sh` | merged Stage 2 checkpoint |
| Stage 2 (0.5B) | `config/train_stage2_0_5B.yaml` | `scripts/train_stage2_0_5B.sh` | exported 0.5B Stage 1 checkpoint |

Launch each stage after updating its YAML:

```bash
bash scripts/train_stage1_7B.sh
bash scripts/train_stage2_7B.sh
bash scripts/train_stage3_7B.sh
```

The 0.5B Stage 2 recipe uses `Qwen2.5-0.5B-Instruct` and a larger per-device batch:

```bash
bash scripts/train_stage2_0_5B.sh
```

YAML values can be overridden on the command line:

```bash
bash scripts/train_stage2_7B.sh \
  --resume_from_checkpoint outputs/train_stage1_7B/exported_model \
  --output_dir outputs/train_stage2_7B \
  --learning_rate 2e-5
```

Use the shared launcher for a custom configuration:

```bash
bash scripts/train.sh config/train_stage2_7B.yaml
```

The shared launcher uses Accelerate by default. Stage 3 selects DeepSpeed with `config/ds_config_zero2.json`; set `FLEXISLM_LAUNCHER` and `DEEPSPEED_CONFIG` to override the launcher or ZeRO configuration. GPU and distributed settings are detected by `scripts/env.sh`.

## Evaluation

FlexiSLM uses the bundled [Kimi-Audio-Evalkit](https://github.com/petrichor20211/Kimi-Audio-Evalkit) submodule to evaluate VoiceBench, OpenAudioBench, and LibriSpeech. Inference and scoring are separate. Run all commands below from the FlexiSLM repository root.

A fresh `--recurse-submodules` clone already contains the Evalkit. For an existing clone, initialize it with `git submodule update --init --recursive`. Then install the Evalkit requirements (not needed for training or inference) and download the benchmark data:

```bash
pip install -r Kimi-Audio-Evalkit/requirements.txt

BENCHMARK_DATA_ROOT="$PWD/data/benchmarks"
python Kimi-Audio-Evalkit/data/download_benchmark.py \
  --datasets VoiceBench,OpenAudioBench,LibriSpeech \
  --output-dir "$BENCHMARK_DATA_ROOT"
```

### 1. Build Requests and Run Inference

```bash
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

Export the DeepSeek API key and run the committed evaluation configuration directly:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
python -m src.eval config/eval_benchmarks.yaml
```

Results are written to `outputs/evaluation/results/`. To run selected jobs only, repeat `--job`, for example:

```bash
python -m src.eval config/eval_benchmarks.yaml \
  --job voicebench_openbookqa \
  --job librispeech_test-clean
```

The committed configuration covers VoiceBench, OpenAudioBench, and LibriSpeech `test-clean`/`test-other` ASR WER. Use `python -m src.eval --help` for all job selection options.

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

