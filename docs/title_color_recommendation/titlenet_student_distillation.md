# TitLeNet Student Distillation

## Purpose

Design a new on-device Student model from the existing TitLeNet ablation study, then train it with Teacher-Student distillation.

The Student is not a direct reuse of `titlenet_fast_b` or any existing deletion-only ablation variant. The ablation results are used as design evidence, and the final Student architecture is registered separately as `titlenet_student`.

## Ablation Evidence

Reference artifacts:

- `outputs/reports/titlenet_ablation_report.md`
- `outputs/reports/titlenet_ablation_results.csv`
- `outputs/reports/titlenet_ablation_latency.png`
- `outputs/reports/titlenet_ablation_ndcg5_delta.png`

Key observations:

| model | params | size_mb | batch1_latency_ms | test_ndcg@5 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| `titlenet` | 183,732 | 0.715 | 2.465 | 0.990784 | Teacher/baseline |
| `titlenet_no_last_extra_residual` | 149,472 | 0.582 | 1.776 | 0.990466 | small quality drop with useful latency gain |
| `titlenet_no_stage1` | 170,328 | 0.662 | 1.858 | 0.990400 | stage simplification is plausible |
| `titlenet_no_last_residual` | 115,212 | 0.448 | 1.418 | 0.989356 | stronger compression with more quality risk |
| `titlenet_no_residual` | 80,048 | 0.311 | 0.777 | 0.986127 | too aggressive for the first Student design |

## Student Architecture

Registered model name:

```text
titlenet_student
```

Design:

```text
Input: [B, 4, 36, 136]
Channels: [32, 64, 96, 128]
Residual blocks: [0, 1, 1]
Attention: ECA
Head hidden dim: 128
Output logits: [B, 32]
```

Design rationale:

- Use narrower channels than the Teacher to reduce parameter count and memory footprint.
- Remove the first residual block and final extra residual depth based on ablation results.
- Keep some residual capacity in the middle and final stages to avoid the larger quality drop seen in full residual removal.
- Use ECA instead of SE for lighter channel attention.
- Reduce classifier head size from 192 to 128.

## Training

Default config:

```text
configs/title_color_recommendation/titlenet_student_distillation.yaml
```

Run Student-only baseline:

```bash
python experiments/title_color_recommendation/run_full_training.py \
  --config configs/title_color_recommendation/full_training.yaml \
  --model-name titlenet_student \
  --activation hardswish \
  --weight-init small_head \
  --best-metric val_ndcg@5 \
  --checkpoint-dir outputs/checkpoints/titlenet_student_only \
  --log-path outputs/logs/titlenet_student_only.jsonl \
  --report-path outputs/reports/model_evaluation/titlenet_student_only_report.md \
  --loss-plot-path outputs/reports/model_evaluation/titlenet_student_only_loss.png \
  --ndcg-plot-path outputs/reports/model_evaluation/titlenet_student_only_ndcg.png \
  --color-plot-path outputs/reports/model_evaluation/titlenet_student_only_colors.png
```

Run Student distillation:

```bash
python experiments/title_color_recommendation/train_titlenet_student_distillation.py \
  --config configs/title_color_recommendation/titlenet_student_distillation.yaml
```

Run the full Student experiment in one command:

```bash
python experiments/title_color_recommendation/run_titlenet_student_experiment.py \
  --config configs/title_color_recommendation/titlenet_student_distillation.yaml
```

This executes:

```text
1. Student-only training
2. Student-distilled fine-tuning from the Student-only best checkpoint
3. Student-only vs Student-distilled comparison report generation
```

## Distillation Loss

The distillation script combines the existing soft label loss with Teacher distribution matching.

```text
total_loss =
  base_loss_weight * soft_label_kl_divergence(student_logits, target_distribution)
  + distillation_loss_weight * KL(student_logits / T, teacher_logits / T) * T^2
```

Default values:

```text
temperature = 2.0
base_loss_weight = 0.8
distillation_loss_weight = 0.2
distillation fine-tune epochs = 10
distillation fine-tune learning_rate = 1e-4
```

The comparison report tracks both `NDCG@3` and `NDCG@5`. Distillation should be reviewed unless both metrics are maintained or improved against the Student-only baseline.

Run the rigorous KD weight sweep:

```bash
python experiments/title_color_recommendation/run_titlenet_student_kd_weight_sweep.py \
  --config configs/title_color_recommendation/titlenet_student_distillation.yaml
```

This trains one Student-only baseline, then evaluates the same KD weight grid for both from-scratch and warm-start distillation:

| phase | base/KD weights | init | epochs | learning rate |
| --- | --- | --- | ---: | ---: |
| KD from scratch | `0.5/0.5`, `0.7/0.3`, `0.8/0.2`, `0.9/0.1` | random | 20 | `5e-4` |
| Warm-start KD | `0.5/0.5`, `0.7/0.3`, `0.8/0.2`, `0.9/0.1` | Student-only best | 10 | `1e-4` |

Sweep outputs:

```text
outputs/reports/model_evaluation/titlenet_student_kd_weight_sweep_report.md
outputs/reports/model_evaluation/titlenet_student_kd_weight_sweep_metrics.json
outputs/reports/model_evaluation/titlenet_student_kd_weight_sweep_results.csv
```

## Quantization Prep

최고 Student 후보인 `warm_start kd_90_10` checkpoint를 양자화하기 전에는 FP32 ONNX baseline을 먼저 고정한다.

```bash
python experiments/title_color_recommendation/prepare_titlenet_student_quantization_baseline.py
```

상세 산출물과 검증 기준은 `docs/title_color_recommendation/titlenet_student_quantization_prep.md`를 따른다.

## Outputs

```text
outputs/checkpoints/titlenet_student_distillation/checkpoint_best.pt
outputs/checkpoints/titlenet_student_distillation/checkpoint_latest.pt
outputs/logs/titlenet_student_distillation.jsonl
outputs/reports/model_evaluation/titlenet_student_distillation_report.md
outputs/reports/model_evaluation/titlenet_student_distillation_metrics.json
outputs/reports/model_evaluation/titlenet_student_experiment_report.md
outputs/reports/model_evaluation/titlenet_student_experiment_metrics.json
```

## Evaluation Criteria

- Student parameter count is lower than Teacher TitLeNet.
- Student model size is lower than Teacher TitLeNet.
- Student batch1 latency is lower than Teacher TitLeNet.
- Student-distilled NDCG@3 is better than or equal to Student-only NDCG@3.
- Student-distilled NDCG@5 is better than or equal to Student-only NDCG@5.
- Student-distilled top-1 agreement with Teacher is high enough for on-device use.
- Final Student logits output remains `[B, 32]`.
- Top-1 deployment export can still wrap logits with `argmax(logits, dim=1)` and output `[B]`.
