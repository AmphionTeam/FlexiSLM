# FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rate

[![arXiv Paper](https://img.shields.io/badge/arXiv_Paper-2606.31247-b31b1b)](https://arxiv.org/abs/2606.31247)
[![demo page](https://img.shields.io/badge/Demo_Page-Github.io-blue)](https://flexislm.github.io)
[![dataset](https://img.shields.io/badge/FlexiSLM_Data_2M-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact)
[![dataset](https://img.shields.io/badge/FlexiSLM_Data_4M-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact)




## About
This repository contains the code for our paper "FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rate". Reproduced data and checkpoints, along with complete guide to train with the data, will be released soon this month.


About our paper: FlexiSLM is the first SLM that supports *dynamic* and *controllable* frame rates on both speech input and output. A single trained model can be steered between 12.5 Hz down to 4.0 Hz without retraining, and its dynamic frame rate mechanism adapts to the varying complexity of speech. Our paper's key contributions include dynamic frame rate SLM framework and validation, accurate and practical frame rate control, and strong quality-efficiency trade-off.

<!-- ![FlexiSLM architecture](assets/flexislm_architecture.png) -->

Overall FlexiSLM architecture is a Thinker-Talker model with dynamic frame-rate compression on speech input and controllable frame-rate generation on speech output.

<!-- The architecture of FlexiSLM is shown in the figure above.  -->

## News
- **August 6, 2026: Data release**. We have released training data resources on HuggingFace, including [**FlexiSLM/FlexiSLM-Data-4M-s2s**](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-4M-s2s), [**FlexiSLM/FlexiSLM-Data-2M-s2s-compact**](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-2M-s2s-compact), [**FlexiSLM/FlexiSLM-Data-5M-t2t**](https://huggingface.co/datasets/FlexiSLM/FlexiSLM-Data-5M-t2t). These data are reproduced based on the paper's data pipeline.
- **August 2, 2026: Code release**. We have released the training and inference code of FlexiSLM-7B.
- Before September 1, 2026: Planned Reproduced FlexiSLM-Data and checkpoint release: We plan to release a reproduced version of FlexiSLM-7B and 5M samples of reproduced speech-to-speech dialog training data. We plan to release them before September 2026. 


## Training Guide

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/AmphionTeam/FlexiSLM.git
cd FlexiSLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
### Repository Layout

```text
FlexiSLM/
├── assets/                 # Static images and other documentation assets
├── config/                 # Declarative training and runtime configurations
│   └── datasets/           # Dataset recipes referenced by training configs
├── examples/               # Small example data and runnable notebooks
│   └── data/               # Minimal ASR, TTS, and dialogue JSONL samples
├── local/                  # Offline data preparation, conversion, and audit tools
├── scripts/                # Thin shell launchers and shared runtime environment setup
├── src/                    # Reusable training, inference, and model implementation
│   ├── dataset/            # Dataset loading, preprocessing, and collation
│   ├── models/             # FlexiSLM model definitions, configs, and loading utilities
│   ├── processor/          # Text and input processing utilities
│   └── trainer/            # Trainer implementation and training helpers
├── README.md               # Installation, training, data, and inference guide
├── LICENSE                 # Project license
└── requirements.txt        # Python dependencies
```

Keep datasets, model checkpoints, training outputs, logs, and temporary files outside the repository. Place reusable runtime code under `src/`; reserve `local/` for offline or corpus-specific utilities, and keep `scripts/` limited to executable shell entrypoints.

### Training Scripts
FlexiSLM training progresses in 3 stages:
1. **Talker pre-training.** Freeze the LLM backbone and train only the randomly initialized Talker end to end on about 100K hours of English TTS. We also add ASR data to pretrain the input merging transformer.
2. **Multi-task LoRA fine-tuning.** Activate the input-side Frame Merging Module, Thinker, and Talker; apply LoRA to the Thinker and train on mixed speech tasks.
3. **Full fine-tuning.** Continue from Stage 2, merge the LoRA updates into the LLM, train all parameters, and enable the Talker-to-Thinker connection to improve speech perception and generation quality.


Training arguments are stored in YAML files under `config/`, while the launch scripts live under `scripts/`:

| Stage | Configuration | Launcher |
| --- | --- | --- |
| Stage 1 | `config/train_stage1.yaml` | `scripts/train_stage1.sh` |
| Stage 2 | `config/train_stage2.yaml` | `scripts/train_stage2.sh` |
| Stage 3 | `config/train_stage3.yaml` | `scripts/train_stage3.sh` |
<!-- | Stage 2 (v1 merging) | `config/train_stage2_v1merging.yaml` | `scripts/train_stage2_v1merging.sh` | -->

The YAML values can be overridden from the command line:

```bash
bash scripts/train_stage2.sh \
  --learning_rate 2e-5 \
  --output_dir outputs/custom_stage2
```

Use the shared launcher to run a custom configuration:

```bash
bash scripts/train.sh config/train_stage2.yaml
```

The scripts detect local or distributed GPU settings through `scripts/env.sh`. They are launch templates, so adjust environment-specific paths and cluster settings before use.

### Dataset Preparation Workflow

Prepare your dataset in three steps.

Step 1: format your JSONL like the provided examples:
- `examples/data/asr.jsonl`
- `examples/data/tts.jsonl`
- `examples/data/dialog.jsonl`

The common JSONL format is:

```json
{
  "messages": [
    {"role": "system", "content": "Respond in a text-audio interleaved manner."},
    {"role": "user", "content": "<|audio|>"},
    {"role": "assistant", "content": "Hello! <|audio|>"}
  ],
  "audios": [
    "/abspath/to/user.wav",
    "/abspath/to/assistant.wav"
  ]
}
```
(system message are always overwritten by Qwen-Omni's system message in our setting. Configure this behavior in src/dataset/interleaved.py)


Important constraints:
- `messages[*].content` containing `<|audio|>` means one audio item should be consumed.
- The total count of `<|audio|>` placeholders must match `len(audios)` for each sample.
- `audios` may use absolute paths, or relative paths resolved from `audio_root` in YAML.

Step 2: use the precompute script to append audio duration/token metadata.

Script path:
- `local/precompute_audio_durations.py`

Single JSONL:

```bash
python local/precompute_audio_durations.py \
  --input path/to/train.jsonl \
  --audio-root path/to/audio_root \
  --workers 32
```

Batch mode (process all `data_paths` listed in a YAML file):

```bash
python local/precompute_audio_durations.py \
  --yaml path/to/train_recipe.yaml \
  --workers 32
```

Output naming:
- `train.jsonl` -> `train.with_durations.jsonl`

Added fields:
- `audio_durations`
- `audio_tokens`
- `num_tokens_est`

Step 3: update YAML recipes to include your dataset files.
Then pass the YAML paths to training via `--dataset_name` and `--dataset_name_eval`.

Example YAML:

```yaml
xlsx_sample_num: 5
audio_root: path/to/audio_root

dataset:
  my_train_set:
    ratio: 1.0
    data_paths:
      - path/to/train.with_durations.jsonl

  my_eval_set:
    ratio: 1.0
    data_paths:
      - path/to/eval.with_durations.jsonl
```

Supported `data_paths` types:
- JSONL file
- WebDataset tar path, directory, or glob (`data_format: webdataset`)
- parquet/local HF dataset path or HF dataset id

For WebDataset, build an index before training so distributed workers do not
scan every tar shard at startup:

```bash
python local/precompute_webdataset_index.py \
  --input 'path/to/shards/*.tar' \
  --output path/to/train.webdataset.jsonl \
  --task tts \
  --prompt-template 'Read the following text out loud: {text}'
```

Then point the recipe at both the tar source and precomputed index:

```yaml
dataset:
  my_webdataset:
    ratio: 1.0
    data_format: webdataset
    webdataset_index_path: path/to/train.webdataset.jsonl
    data_paths:
      - path/to/shards/*.tar
```

The index stores `wds://<tar>::<member>#ch=<channel>` references. Audio remains
inside the tar shards and is decoded on demand during training.

Run a one-step Stage 2 smoke test against the local S2S WebDataset with:

```bash
bash scripts/debug_stage2_s2s_webdataset.sh
```

The launcher indexes four samples from the first training shard, uses one GPU
by default, disables remote reporting and checkpoint saving, and runs a single
optimizer step. Override `S2S_DATA_ROOT` or `S2S_DEBUG_SHARD` when needed.

## Inference Guide

Primary inference script: `src/inference_flexislm.py`.

Use:

```bash
python -m src.inference_flexislm --help
```

Minimal API examples:

```python
from src.inference_flexislm import InterleavedInferenceConfig, InterleavedS2SInference

cfg = InterleavedInferenceConfig(
    model_path="/path/to/flexislm_checkpoint",
    use_flow_matching_decoder=False,
    flexicodec_ckpt_path="/path/to/flexicodec_ckpt.safetensors",
    flexicodec_config_path="/path/to/flexicodec_config.yaml",
    sensevoice_path="/path/to/SenseVoiceSmall",
)
engine = InterleavedS2SInference(cfg, device="cuda")

debug_sentence = "And henry the eighth appropriated to himself the religious house of grey ladies and all the properties appertaining thereto."
debug_audio_path = "/path/to/input_audio.wav"

# Text-to-Speech Synthesis
tts = engine.generate_tts(
    sentence=debug_sentence,
    framerate=1.0,
)

# Text-to-Speech
t2s = engine.generate_from_text(
    text_input=f"Please read the following text: {debug_sentence}",
    history="",
    framerate=1.0,
    output_text_only=False,
)

# Speech-to-Speech
s2s = engine.generate_from_audio(
    audio_path=debug_audio_path,
    text_query="Please respond naturally to the audio in speech.",
    history="",
    framerate=1.0,
    output_text_only=False,
)

# Speech-to-Text
s2t = engine.generate_from_audio(
    audio_path=debug_audio_path,
    text_query="Please transcribe the speech in the audio.",
    history="",
    framerate=1.0,
    output_text_only=True,
)

# Text-to-Text
t2t = engine.generate_from_text(
    text_input="What is dynamic frame rate in speech modeling? Answer in one sentence.",
    history="",
    framerate=1.0,
    output_text_only=True,
)
```

Quick debug run (same five modes in script):

```bash
python -m src.inference_flexislm \
  --model_path /path/to/flexislm_checkpoint \
  --debug \
  --debug_audio_path /path/to/input_audio.wav
```

Minimal notebook example (imports inference module and runs T2T/S2T/TTS):

```bash
examples/inference.ipynb
```

## Citation and Acknowledgements
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

Acknowledgements:
- Our work uses Qwen 2.5 as the backbone and [Qwen 2.5-Omni](https://github.com/qwenlm/qwen2.5-omni) as audio encoder. 
- Our training framework is largely based on Huggingface [Transformers](https://github.com/huggingface/transformers).
- Our previous open-source works [FlexiCodec](https://github.com/AmphionTeam/FlexiCodec) and [DualCodec](https://github.com/jiaqili3/DualCodec) are foundational to this work.


## License

This project is licensed under the MIT License. 
