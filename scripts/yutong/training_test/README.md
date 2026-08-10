# WebDataset training test

该脚本使用当前生产 WebDataset 流程进行 4 卡、300 step 训练测试。滚动补池和 metadata duration 提前过滤是固定实现，不需要环境变量或策略配置。

```bash
bash /F00120260003/flexislm_project/FlexiSLM/scripts/yutong/training_test/run_webdataset.sh
```

内部固定配置：

```text
RUN_NAME=webdataset_optimized_4gpu_v1
AB_STEPS=300
CUDA_VISIBLE_DEVICES=0,1,2,3
EXPECTED_NUM_GPUS=4
```

结果保存在：

```text
/F00120260003/flexislm_project/yutong/outputs/webdataset_ab/webdataset_optimized_4gpu_v1/
```

输出包括：

- `ab_metadata.env`：实验固定配置；
- `train.log`：完整 stdout/stderr；
- `profiler/rank*/`：各 rank 的 CPU/CUDA/NCCL trace；
- `webdataset_quarantine.rank*.worker*.jsonl`：需要隔离的解码异常。

重点指标：

- step time / samples per second；
- `webdataset/main_loader_wait_time`；
- `webdataset/main_loader_wait_over_*_ratio`；
- `webdataset/samples_filtered_duration`；
- `webdataset/data_wait_time`；
- GPU utilization 和 profiler 中的 NCCL 等待区间。
