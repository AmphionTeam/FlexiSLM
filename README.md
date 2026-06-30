# FlexiSLM: A Dynamic and Controllable Frame Rate Spoken Language Model

[![demo page](https://img.shields.io/badge/demo-page-blue)](https://flexislm.github.io)



## About
Spoken language models (SLMs) extend LLMs to speech input and output.
Recent audio tokenizer research has proposed dynamic frame rate speech coding, which
exploits this non-uniformity and enables two new capabilities: very low average
frame rates and frame rate controllability. However, this technique has not yet
been applied to SLMs.


We introduce Flexible Spoken Language Model (FlexiSLM),
the first SLM that supports dynamic and controllable frame rates on both speech
input and output. Using dynamic frame rate representations, FlexiSLM
outperforms fixed-frame-rate 7B models including Qwen2.5-Omni and Kimi-Audio at
its high-quality operating points. We further verify that FlexiSLM can be
accurately steered down to 4.0 Hz; at 6.25 Hz, it roughly halves inference time
relative to 12.5 Hz while retaining strong speech-to-speech quality.

## Status
- Code release: We hope to release training and inference code by Auguest 1st, 2026. We are waiting for approval.
- Reproduced FlexiSLM-Data and checkpoint release: We plan to release a reproduced version of FlexiSLM-7B. We plan to release them before September 2026.
