# FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rate

[![arXiv Paper](https://img.shields.io/badge/arXiv_Paper-2606.31247-b31b1b)](https://arxiv.org/abs/2606.31247)
[![demo page](https://img.shields.io/badge/Demo_Page-Github.io-blue)](https://flexislm.github.io)
[![dataset](https://img.shields.io/badge/Data-2M_speech2speech-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact)
[![dataset](https://img.shields.io/badge/Data-4M_speech2speech-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s)

## Overview

This repository contains the code for our paper, "FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rate."

FlexiSLM is the first spoken language model that supports *dynamic* and *controllable* frame rates on both speech input and output. A single trained model can be steered between 12.5 Hz and 4.0 Hz without retraining, while its dynamic frame-rate mechanism adapts to the varying complexity of speech. FlexiSLM uses a Thinker-Talker architecture with dynamic frame-rate compression on speech input and controllable frame-rate generation on speech output.

<!-- ![FlexiSLM architecture](assets/flexislm_architecture.png) -->

## News

- **August 20, 2026: Checkpoint release.** We released the reproduced [FlexiSLM-7B Stage 2 checkpoint](https://huggingface.co/FlexiSLM/FlexiSLM-7B-Stage2-v1).
- **August 6, 2026: Data release.** We released [FlexiSLM-Data-4M-s2s](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s), [FlexiSLM-Data-2M-s2s-compact](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact), and [FlexiSLM-Data-5M-t2t](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-5M-t2t).
- **August 2, 2026: Code release.** We released the FlexiSLM-7B training and inference code.

## Project Structure

```text
FlexiSLM/
├── assets/                 # Documentation and demo assets
├── config/                 # Training and dataset YAML configurations
│   └── datasets/           # Dataset recipes used by training
├── examples/               # Inference notebook and small examples
├── local/                  # Data conversion and benchmark request tools
├── scripts/                # Training launchers and runtime setup
├── src/                    # Model, training, inference, and evaluation code
│   ├── dataset/            # Dataset loading and collation
│   ├── eval/               # Kimi-Audio-Evalkit adapters
│   ├── infer/              # YAML-driven inference runner
│   ├── models/             # FlexiSLM and vendored FlexiCodec implementation
│   └── trainer/            # Trainer implementation
├── README.md
├── LICENSE
└── requirements.txt
```

## Inference

The released 7B checkpoint supports TTS, ASR, audio question answering, and speech-to-speech generation.

### 1. Installation

```bash
git clone https://github.com/AmphionTeam/FlexiSLM.git
cd FlexiSLM
pip install -r requirements.txt
```

### 2. Model Download

```bash
MODEL_ROOT=/path/to/models
mkdir -p "$MODEL_ROOT/FlexiCodec"

hf download FlexiSLM/FlexiSLM-7B-Stage2-v1 \
  --local-dir "$MODEL_ROOT/FlexiSLM-7B-Stage2-v1"
hf download Qwen/Qwen2.5-Omni-7B \
  --local-dir "$MODEL_ROOT/Qwen2.5-Omni-7B"
hf download iic/SenseVoiceSmall \
  --local-dir "$MODEL_ROOT/SenseVoiceSmall"
hf download openai/whisper-large-v3 \
  --local-dir "$MODEL_ROOT/whisper-large-v3"
hf download jiaqili3/flexicodec \
  12hz_v1_half_config.yaml nartts_flexicodec_only.safetensors \
  --local-dir "$MODEL_ROOT/FlexiCodec"
```

### 3. Python API

For a single request or interactive use, call the inference engine directly without creating a JSONL file:

```python
from pathlib import Path

import soundfile as sf
import torch

from src.inference_flexislm import (
    InterleavedInferenceConfig,
    InterleavedS2SInference,
)

config = InterleavedInferenceConfig(
    model_path="/path/to/models/FlexiSLM-7B-Stage2-v1",
    qwen25o_encoder_path="/path/to/models/Qwen2.5-Omni-7B",
    qwen25o_encoder_config_path="/path/to/models/Qwen2.5-Omni-7B/config.json",
    flexicodec_ckpt_path="/path/to/models/FlexiCodec/nartts_flexicodec_only.safetensors",
    flexicodec_config_path="/path/to/models/FlexiCodec/12hz_v1_half_config.yaml",
    sensevoice_path="/path/to/models/SenseVoiceSmall",
    use_flow_matching_decoder=False,
    enable_flexible_framerate=True,
    input_framerate=8.0,
    input_base_rate=12.5,
    default_framerate=8.0,
    decode_audio=True,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
engine = InterleavedS2SInference(config, device="cuda:0")


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
    audio_path="/path/to/input.wav",
    text_query="Please transcribe the audio.",
    framerate=8.0,
    output_text_only=True,
)
print(result["text"])

# Audio question answering
result = engine.generate_from_audio(
    audio_path="/path/to/question.wav",
    text_query="",
    framerate=8.0,
    output_text_only=True,
)
print(result["text"])

# Speech-to-speech generation
result = engine.generate_from_audio(
    audio_path="/path/to/input.wav",
    text_query="",
    framerate=8.0,
    output_text_only=False,
)
save_audio(result, "s2s.wav")
```

A minimal notebook is available at `examples/inference.ipynb`.

### 4. Batch Inference

Batch inference reads requests from JSONL and uses a YAML file for model, input, output, and multi-GPU runtime settings. Create `/path/to/requests.jsonl`:

```jsonl
{"index": 0, "task": "tts", "input": {"text": "FlexiSLM supports controllable speech generation."}, "metadata": {"sample_id": "tts-demo"}}
{"index": 1, "task": "asr", "input": {"audio_path": "/path/to/input.wav"}, "metadata": {"sample_id": "asr-demo"}}
{"index": 2, "task": "audio_qa", "input": {"audio_path": "/path/to/question.wav", "model_prompt": ""}, "metadata": {"sample_id": "qa-demo"}}
{"index": 3, "task": "s2s", "input": {"audio_path": "/path/to/input.wav"}, "metadata": {"sample_id": "s2s-demo"}}
```

Create `/path/to/infer_7b.yaml` and replace the paths:

```yaml
engine:
  config:
    model_path: /path/to/models/FlexiSLM-7B-Stage2-v1
    qwen25o_encoder_path: /path/to/models/Qwen2.5-Omni-7B
    qwen25o_encoder_config_path: /path/to/models/Qwen2.5-Omni-7B/config.json
    flexicodec_ckpt_path: /path/to/models/FlexiCodec/nartts_flexicodec_only.safetensors
    flexicodec_config_path: /path/to/models/FlexiCodec/12hz_v1_half_config.yaml
    sensevoice_path: /path/to/models/SenseVoiceSmall
    use_flow_matching_decoder: false
    enable_flexible_framerate: true
    input_framerate: 8.0
    input_base_rate: 12.5
    default_framerate: 8.0
    decode_audio: true
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2

input:
  path: /path/to/requests.jsonl

output:
  trace_path: /path/to/inference/traces.jsonl
  audio_dir: /path/to/inference/audio
  error_path: /path/to/inference/errors.jsonl

inference:
  checkpoint: /path/to/models/FlexiSLM-7B-Stage2-v1
  target_framerate_hz: 8.0
  transcribe_model_path: /path/to/models/whisper-large-v3
  output_sample_rate: 16000

runtime:
  devices: [cuda:0]
  workers_per_device: 1
  fail_fast: false
```

Run the batch inference entrypoint:

```bash
python -m src.infer /path/to/infer_7b.yaml
```

The runner writes one unified JSONL trace and stores generated speech under `output.audio_dir`. `inference.transcribe_model_path` is a batch-level setting used to transcribe generated `s2s` audio for evaluation. It can be omitted when transcription is not needed.

## Data

We open-source the data produced by the following pipeline:

1. **Prompt collection and response generation.** Text prompts are collected from public QA, instruction-following, and dialogue datasets. Responses are generated with Qwen3-Omni-30B-A3B. The resulting text pairs are released as [FlexiSLM-Data-5M-t2t](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-5M-t2t).
2. **Speech synthesis.** Responses are synthesized with Qwen3-TTS, while prompts are synthesized with Fish-Audio using randomly sampled speaker prompts. The resulting 4.2M samples and approximately 26K hours of audio are released as [FlexiSLM-Data-4M-s2s](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s).
3. **Quality filtering and compression.** Stricter filtering is applied and all audio is converted to MP3. The compact release contains 2.43M samples and approximately 14.8K hours of audio in about 385 GB: [FlexiSLM-Data-2M-s2s-compact](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact).

Download the compact training set with:

```bash
DATA_ROOT=/path/to/data
hf download FlexiSLM/FlexiSLM-Data-2M-s2s-compact \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/FlexiSLM-Data-2M-s2s-compact"
```

The compact release uses native WebDataset shards. Each sample contains a question MP3, a response MP3, and JSON metadata:

```text
00000001.question.mp3
00000001.response.mp3
00000001.json
```

Training shards follow this pattern:

```text
/path/to/data/FlexiSLM-Data-2M-s2s-compact/data/train-{00000..00242}-of-00243.tar
```

## Training

FlexiSLM training has three stages:

1. **Talker and input-module pre-training.** Freeze the Qwen backbone and train the Talker, audio embeddings, and input frame-merging module.
2. **Multi-task LoRA fine-tuning.** Train the Talker and input modules while adapting the Thinker with LoRA.
3. **Full fine-tuning.** Merge the Stage 2 LoRA weights into the Thinker, enable the Talker-to-Thinker connection, and train all model components.

### 1. Installation

```bash
git clone https://github.com/AmphionTeam/FlexiSLM.git
cd FlexiSLM
pip install -r requirements.txt

MODEL_ROOT=/path/to/models
mkdir -p "$MODEL_ROOT/FlexiCodec"
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen2.5-7B-Instruct"
hf download Qwen/Qwen2.5-Omni-7B \
  --local-dir "$MODEL_ROOT/Qwen2.5-Omni-7B"
hf download iic/SenseVoiceSmall \
  --local-dir "$MODEL_ROOT/SenseVoiceSmall"
hf download jiaqili3/flexicodec \
  12hz_v1_half_config.yaml nartts_flexicodec_only.safetensors \
  --local-dir "$MODEL_ROOT/FlexiCodec"
```

In each of `config/train_stage1.yaml`, `config/train_stage2.yaml`, and `config/train_stage3.yaml`, replace the `/path/to/models/...` values for:

- `model_name_or_path`, `config_name`, and `tokenizer_name`
- `qwen25omni_encoder_path` and `qwen25omni_encoder_config_path`
- `flexicodec_config_path` and `flexicodec_ckpt_path`
- `sensevoice_small_path`

Stage 2 additionally requires `resume_from_checkpoint` to point to the exported Stage 1 model. Stage 3 requires `resume_from_checkpoint` to point to a Stage 2 checkpoint whose LoRA weights have already been merged.

### 2. Data Configuration

Download a released dataset as described in [Data](#data), then edit:

- `config/datasets/train_stage1.yaml` for Stage 1
- `config/datasets/train_stage2_3.yaml` for Stages 2 and 3

Replace the shard path under `dataset.speech2speech.data_paths`:

```yaml
dataset_backend: webdataset_stream

dataset:
  speech2speech:
    ratio: 1.0
    data_format: webdataset_stream
    data_paths:
      - /path/to/data/FlexiSLM-Data-2M-s2s-compact/data/train-{00000..00242}-of-00243.tar
    webdataset:
      layout: s2s_pair
```

The provided recipe uses 60,645 batches per epoch, corresponding to 2,425,778 training samples with 8 GPUs and 5 samples per GPU. Adjust `webdataset_runtime.sampling.steps_per_epoch` when changing the GPU count or per-device batch size.

### 3. Launch Training

Training arguments are stored in YAML files under `config/`; launchers live under `scripts/`:

| Stage | Configuration | Launcher | Initialization |
| --- | --- | --- | --- |
| Stage 1 | `config/train_stage1.yaml` | `scripts/train_stage1.sh` | Qwen2.5-7B base model |
| Stage 2 | `config/train_stage2.yaml` | `scripts/train_stage2.sh` | exported Stage 1 checkpoint |
| Stage 3 | `config/train_stage3.yaml` | `scripts/train_stage3.sh` | merged Stage 2 checkpoint |

Launch each stage after updating its YAML:

```bash
bash scripts/train_stage1.sh
bash scripts/train_stage2.sh
bash scripts/train_stage3.sh
```

YAML values can be overridden on the command line:

```bash
bash scripts/train_stage2.sh \
  --resume_from_checkpoint /path/to/stage1_checkpoint \
  --output_dir /path/to/outputs/stage2 \
  --learning_rate 2e-5
```

Use the shared launcher for a custom configuration:

```bash
bash scripts/train.sh /path/to/train.yaml
```

The shared launcher uses Accelerate by default. Stage 3 selects DeepSpeed with `config/ds_config_zero2.json`; set `FLEXISLM_LAUNCHER` and `DEEPSPEED_CONFIG` to override the launcher or ZeRO configuration. GPU and distributed settings are detected by `scripts/env.sh`.

Training does not construct an evaluation dataset. Run inference with `src.infer`, then evaluate its traces with `src.eval`.

## Evaluation

FlexiSLM uses [Kimi-Audio-Evalkit](https://github.com/MoonshotAI/Kimi-Audio-Evalkit) for VoiceBench, OpenAudioBench, ASR, and TTS metrics. Inference and scoring are separate so GPU generation does not wait for LLM judges.

### 1. Setup

```bash
git clone https://github.com/MoonshotAI/Kimi-Audio-Evalkit.git
cd Kimi-Audio-Evalkit
git submodule update --init --recursive
pip install -r requirements.txt
```

Follow `Kimi-Audio-Evalkit/data/README.md` to download the desired benchmarks and set `DATASETS.dataset_root` in its `config.yaml`.

### 2. Inference

For VoiceBench or OpenAudioBench, build FlexiSLM requests from the Evalkit data root:

```bash
cd /path/to/FlexiSLM
python local/build_vb_oab_requests.py \
  --benchmark voicebench \
  --data-root /path/to/evalkit/data \
  --out-dir /path/to/requests/voicebench \
  --task audio_qa
```

Run each generated request file through `python -m src.infer` using an inference YAML like the one in [Inference](#inference). Keep one `traces.jsonl` per benchmark subset.

### 3. Scoring

Create an evaluation YAML such as `/path/to/eval_voicebench.yaml`:

```yaml
evalkit_path: /path/to/Kimi-Audio-Evalkit
data_root: /path/to/evalkit/data
model_name: FlexiSLM-7B-Stage2-v1
judge_model: gpt-4o-mini

jobs:
  - name: voicebench_openbookqa
    benchmark: voicebench
    dataset: openbookqa
    method: vb-mcq
    trace_file: /path/to/inference/openbookqa/traces.jsonl
    result_path: /path/to/evaluation/openbookqa_performance.json
```

Then run:

```bash
cd /path/to/FlexiSLM
python -m src.eval /path/to/eval_voicebench.yaml
```

Rule-based subsets do not require an API key. LLM-judged subsets such as `alpacaeval_full`, `commoneval`, and OpenAudioBench require an OpenAI-compatible judge credential:

```bash
export OPENAI_API_KEY=your_api_key
```

Add one job per trace file. Supported `benchmark` values are `voicebench`, `openaudiobench`, `asr`, and `tts`; use `python -m src.eval --help` for job selection options.

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
- Our previous open-source works [FlexiCodec](https://github.com/AmphionTeam/FlexiCodec) and [DualCodec](https://github.com/jiaqili3/DualCodec) are foundational to this work.

## License

This project is licensed under the MIT License.
