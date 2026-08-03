# FlexiSLM: A Spoken Language Model with Dynamic and Controllable Frame Rate

[![arXiv](https://img.shields.io/badge/arXiv-2606.31247-b31b1b)](https://arxiv.org/abs/2606.31247)
[![demo page](https://img.shields.io/badge/demo-page-blue)](https://flexislm.github.io)



## About
This repository contains the code for FlexiSLM paper. Reproduced data and checkpoints, along with complete guide to train with the data, will be released soon this month.


About FlexiSLM: FlexiSLM is the first SLM that supports *dynamic* and *controllable* frame rates on both speech input and output. A single trained model can be steered between 12.5 Hz down to 4.0 Hz without retraining, and its dynamic frame rate mechanism adapts to the varying complexity of speech.
Key contributions include: 
- **Dynamic frame rate SLM framework and validation.** We introduce FlexiSLM, the first dynamic frame rate SLM framework, with dynamic frame compression on both speech input and output. Experiments show strong performance at 12.5 Hz and 6.25 Hz, with graceful degradation at 5.0 Hz and 4.0 Hz.
- **Accurate and practical frame rate control.** We propose direct frame rate conditioning, letting users specify the average output frame rate instead of indirectly tuning a merging threshold. This makes FlexiSLM, to our knowledge, the first SLM with frame rate controllability.
- **Strong quality-efficiency trade-off.** At 6.25 Hz output, FlexiSLM roughly *halves* AR inference time relative to 12.5 Hz with only minor quality degradation; at high-quality operating points, it outperforms fixed-rate 7B baselines such as Qwen2.5-Omni and Kimi-Audio.

<!-- ![FlexiSLM architecture](assets/flexislm_architecture.png) -->

Overall FlexiSLM architecture is a Thinker-Talker model with dynamic frame-rate compression on speech input and controllable frame-rate generation on speech output.

<!-- The architecture of FlexiSLM is shown in the figure above.  -->

## News
- **August 2, 2026: Code release**. We have released the training and inference code of FlexiSLM-7B.
- Before September 1, 2026: Planned Reproduced FlexiSLM-Data and checkpoint release: We plan to release a reproduced version of FlexiSLM-7B and 5M samples of reproduced speech-to-speech dialog training data. We plan to release them before September 2026. 



## Training Guide


### Environment Setup

This repository depends on a git submodule: `flexislm/third_party/flexicodec`.

1. Clone with submodules:
```bash
git clone --recurse-submodules <repo_url>
cd flexislm_opensource
```

2. If you already cloned without submodules:
```bash
git submodule update --init --recursive
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 1) Which scripts to use
FlexiSLM training progresses in 3 stages:
1. **Talker pre-training.** Freeze the LLM backbone and train only the randomly initialized Talker end to end on about 100K hours of English TTS. We also add ASR data to pretrain the input merging transformer.
2. **Multi-task LoRA fine-tuning.** Activate the input-side Frame Merging Module, Thinker, and Talker; apply LoRA to the Thinker and train on mixed speech tasks.
3. **Full fine-tuning.** Continue from Stage 2, merge the LoRA updates into the LLM, train all parameters, and enable the Talker-to-Thinker connection to improve speech perception and generation quality.


Use scripts under `exps/` as launch templates:

- `exps/train_stage1.sh`: stage-1 training
- `exps/train_stage2.sh`: stage-2 training
<!-- - `exps/train_stage2_v2merging.sh`: stage-2 with input-merging-v2 options -->
- `exps/train_stage3.sh`: stage-3 training
<!-- - `exps/train_stage3_v2merging.sh`: wrapper that calls `train_stage2_v2merging.sh` -->

Notes:
- These scripts are environment-specific templates and contain cluster/internal absolute paths.
<!-- - Copy a script and edit paths before use.
- If an old script still has deprecated args (for example `--load_from_stage1`), remove them. -->

<!-- ### 2) Recommended direct launch (minimal)

Run from repo root (`flexislm_opensource/`):

```bash
torchrun --nproc_per_node=8 train.py \
  --do_train \
  --log_level info \
  --model_name_or_path path/to/base_qwen_checkpoint \
  --config_name path/to/base_qwen_checkpoint \
  --tokenizer_name path/to/base_qwen_checkpoint \
  --dataset_name flexislm/dataset/recipes/dataset_train_stage1.yaml \
  --dataset_name_eval flexislm/dataset/dataset_eval.yaml \
  --output_dir outputs/flexislm_stage1 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --model_max_length 1024 \
  --learning_rate 2e-5 \
  --bf16 True \
  --torch_dtype bfloat16 \
  --use_parallel True \
  --enable_flexible_framerate True \
  --use_omni_token True
```

Resume training:

```bash
torchrun --nproc_per_node=8 train.py \
  --do_train \
  --resume_from_checkpoint path/to/checkpoint-dir \
  --dataset_name path/to/train_recipe.yaml \
  --dataset_name_eval path/to/eval_recipe.yaml
``` -->

### 3) Dataset Preparation Workflow (JSONL -> durations -> YAML)

Prepare your dataset in three steps.

Step 1: format your JSONL like the provided examples:
- `flexislm/dataset/example_data_asr.jsonl`
- `flexislm/dataset/example_data_tts.jsonl`
- `flexislm/dataset/example_data_dialog.jsonl`

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
(system message are always overwritten by Qwen-Omni's system message in our setting. Configure this behavior in flexislm/dataset/dataset_override/dataset_interleaved.py)


Important constraints:
- `messages[*].content` containing `<|audio|>` means one audio item should be consumed.
- The total count of `<|audio|>` placeholders must match `len(audios)` for each sample.
- `audios` may use absolute paths, or relative paths resolved from `audio_root` in YAML.

Step 2: use the precompute script to append audio duration/token metadata.

Script path:
- `flexislm/dataset/scripts/precompute_audio_durations.py`

Single JSONL:

```bash
python flexislm/dataset/scripts/precompute_audio_durations.py \
  --input path/to/train.jsonl \
  --audio-root path/to/audio_root \
  --workers 32
```

Batch mode (process all `data_paths` listed in a YAML file):

```bash
python flexislm/dataset/scripts/precompute_audio_durations.py \
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
- local directory / file containing WebDataset `.tar` shards (`data_format: webdataset`)
- parquet/local HF dataset path or HF dataset id

### 4) Inference entry

Primary inference script: `flexislm/inference_flexislm.py`.

Use:

```bash
python flexislm/inference_flexislm.py --help
```

Minimal API examples (TTS -> T2S -> S2S -> S2T -> T2T):

```python
from flexislm.inference_flexislm import InterleavedInferenceConfig, InterleavedS2SInference

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

# 1) TTS
tts = engine.generate_tts(
    sentence=debug_sentence,
    framerate=1.0,
)

# 2) T2S
t2s = engine.generate_from_text(
    text_input=f"Please read the following text: {debug_sentence}",
    history="",
    framerate=1.0,
    output_text_only=False,
)

# 3) S2S
s2s = engine.generate_from_audio(
    audio_path=debug_audio_path,
    text_query="Please respond naturally to the audio in speech.",
    history="",
    framerate=1.0,
    output_text_only=False,
)

# 4) S2T
s2t = engine.generate_from_audio(
    audio_path=debug_audio_path,
    text_query="Please transcribe the speech in the audio.",
    history="",
    framerate=1.0,
    output_text_only=True,
)

# 5) T2T
t2t = engine.generate_from_text(
    text_input="What is dynamic frame rate in speech modeling? Answer in one sentence.",
    history="",
    framerate=1.0,
    output_text_only=True,
)
```

Quick debug run (same five modes in script):

```bash
python flexislm/inference_flexislm.py \
  --model_path /path/to/flexislm_checkpoint \
  --debug \
  --debug_audio_path /path/to/input_audio.wav
```

Minimal notebook example (imports inference module and runs T2T/S2T/TTS):

```bash
inference_minimal.ipynb
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
