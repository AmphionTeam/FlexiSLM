# FlexiSLM: A Dynamic and Controllable Frame Rate Spoken Language Model

[![arXiv](https://img.shields.io/badge/arXiv-2606.31247-b31b1b)](https://arxiv.org/abs/2606.31247)
[![demo page](https://img.shields.io/badge/demo-page-blue)](https://flexislm.github.io)


## Project Status
- Code release: We hope to release training and inference code by Auguest 1st, 2026. We are waiting for approval for releasing.
- Reproduced FlexiSLM-Data and checkpoint release: We plan to release a reproduced version of FlexiSLM-7B. We plan to release them before September 2026. 

## About
Existing spoken language models (SLMs) typically use a fixed speech-token frame rate (for example, 25 Hz or 12.5 Hz). This fixed-rate design cannot adapt to time-varying speech complexity and does not offer a direct speed-quality trade-off at inference time. We introduce **FlexiSLM**, the first SLM that supports *dynamic* and *controllable* frame rates on both speech input and output. A single trained model can be steered from 12.5 Hz down to 4.0 Hz without retraining.

### Key contributions
- **Dynamic frame rate SLM framework and validation.** We introduce FlexiSLM, the first dynamic frame rate SLM framework, with dynamic frame compression on both speech input and output. Experiments show strong performance at 12.5 Hz and 6.25 Hz, with graceful degradation at 5.0 Hz and 4.0 Hz. We plan to release the code and model to support future research.
- **Accurate and practical frame rate control.** We propose direct frame rate conditioning, letting users specify the average output frame rate instead of indirectly tuning a merging threshold. This makes FlexiSLM, to our knowledge, the first SLM with frame rate controllability.
- **Strong quality-efficiency trade-off.** At 6.25 Hz output, FlexiSLM roughly *halves* AR inference time relative to 12.5 Hz with only minor quality degradation; at high-quality operating points, it outperforms fixed-rate 7B baselines such as Qwen2.5-Omni and Kimi-Audio.

![FlexiSLM architecture](assets/flexislm_architecture.png)

Overall FlexiSLM architecture: a Thinker-Talker model with dynamic frame-rate compression on speech input and controllable frame-rate generation on speech output.

The architecture of FlexiSLM is shown in the figure above. Its training progresses in 3 stages:
1. **Talker pre-training.** Freeze the LLM backbone and train only the randomly initialized Talker end to end on about 100K hours of English TTS.
2. **Multi-task LoRA fine-tuning.** Activate the input-side Frame Merging Module, Thinker, and Talker; apply LoRA to the Thinker and train on mixed speech tasks.
3. **Full fine-tuning.** Continue from Stage 2, merge the LoRA updates into the LLM, train all parameters, and enable the Talker-to-Thinker connection to improve speech perception and generation quality.

