# TitLeNet Hyperparameter and Initialization Summary

## 목적

이 문서는 최종 모델인 TitLeNet에 직접 연결되는 하이퍼파라미터 및 가중치 초기화 실험만 정리한다.

정리 범위:

```text
titlenet
simple_cnn_m_res_se
```

`simple_cnn_m_res_se`는 현재 코드에서 `titlenet`으로 등록된 최종 모델 구조다. 이 문서와 CSV에는 최종 모델에 직접 연결되는 결과만 남겼다.

## 최종 선택값

| 항목 | 최종값 |
| --- | --- |
| model_name | `titlenet` |
| loss | `KLDivLoss(batchmean)` |
| optimizer | `AdamW` |
| scheduler | `cosine` |
| learning_rate | `0.0005` |
| weight_decay | `0.0001` |
| batch_size | `64` |
| epochs | `20` |
| dropout | `0.2` |
| activation | `gelu` |
| weight_init | `small_head` |
| best_metric | `val_ndcg@5` |
| seed | `42` |

최종 full training 결과:

| run | best_epoch | best_val_loss | best_val_ndcg@3 | best_val_ndcg@5 | test_loss | test_ndcg@3 | test_ndcg@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `titlenet_ndcg3_eval` | 11 | 0.006219 | 0.988420 | 0.990426 | 0.006222 | 0.988514 | 0.990352 |

## TitLeNet 후보 실험

| trial | model | lr | wd | dropout | activation | init | best_epoch | val_loss | val_ndcg@5 | max_color_share |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `simple_cnn_m_res_se_gelu` | `simple_cnn_m_res_se` | 0.0005 | 0.0005 | 0.3 | GELU | `small_head` | 19 | 0.005942 | 0.990176 | 0.399385 |
| `simple_cnn_m_res_se_gelu_lr3e-4` | `simple_cnn_m_res_se` | 0.0003 | 0.0005 | 0.3 | GELU | `small_head` | 19 | 0.006658 | 0.988740 | 0.395211 |
| `simple_cnn_m_res_se_gelu_wd1e-4_drop0.2` | `simple_cnn_m_res_se` | 0.0005 | 0.0001 | 0.2 | GELU | `small_head` | 19 | 0.005802 | 0.990598 | 0.396529 |
| `simple_cnn_m_res_se_hardswish` | `simple_cnn_m_res_se` | 0.0005 | 0.0005 | 0.3 | Hardswish | `small_head` | 19 | 0.006320 | 0.990045 | 0.398286 |
| `titlenet_ndcg3_eval` | `titlenet` | 0.0005 | 0.0001 | 0.2 | GELU | `small_head` | 11 | 0.006219 | 0.990426 | 0.389499 |

## 하이퍼파라미터 판단

### Learning Rate

| 비교 | 결과 |
| --- | --- |
| `lr=0.0005` | best `val_ndcg@5 = 0.990598` |
| `lr=0.0003` | best `val_ndcg@5 = 0.988740` |

`0.0005`가 TitLeNet 후보군에서 더 좋은 validation ranking을 보였다.

### Weight Decay와 Dropout

| wd | dropout | val_loss | val_ndcg@5 |
| ---: | ---: | ---: | ---: |
| 0.0005 | 0.3 | 0.005942 | 0.990176 |
| 0.0001 | 0.2 | 0.005802 | 0.990598 |

`weight_decay=0.0001`, `dropout=0.2` 조합이 더 좋았다.

### Activation

| activation | val_loss | val_ndcg@5 |
| --- | ---: | ---: |
| GELU | 0.005802 | 0.990598 |
| Hardswish | 0.006320 | 0.990045 |

GELU가 Hardswish보다 validation loss와 NDCG 모두에서 우세했다.

## 가중치 초기화 판단

TitLeNet 관련 실험은 모두 `small_head` 초기화를 사용했다. 따라서 최종 TitLeNet 범위 안에서는 초기화 방식 간 직접 비교가 아니라, `small_head`를 적용한 TitLeNet 후보들의 하이퍼파라미터 비교가 수행된 것이다.

| trial | init | val_loss | val_ndcg@5 |
| --- | --- | ---: | ---: |
| `simple_cnn_m_res_se_gelu` | `small_head` | 0.005942 | 0.990176 |
| `simple_cnn_m_res_se_gelu_lr3e-4` | `small_head` | 0.006658 | 0.988740 |
| `simple_cnn_m_res_se_gelu_wd1e-4_drop0.2` | `small_head` | 0.005802 | 0.990598 |
| `simple_cnn_m_res_se_hardswish` | `small_head` | 0.006320 | 0.990045 |
| `titlenet_ndcg3_eval` | `small_head` | 0.006219 | 0.990426 |

최종 모델 문서에서는 `small_head`만 TitLeNet 관련 초기화 결과로 유지한다.

## 최종 결론

최종 TitLeNet 설정은 다음과 같이 유지한다.

```text
model_name: titlenet
learning_rate: 0.0005
weight_decay: 0.0001
dropout: 0.2
activation: gelu
weight_init: small_head
batch_size: 64
scheduler: cosine
best_metric: val_ndcg@5
```

선택 근거:

- `lr=5e-4`가 `lr=3e-4`보다 좋았다.
- `wd=1e-4`, `dropout=0.2`가 가장 높은 validation NDCG를 냈다.
- GELU가 Hardswish보다 좋았다.
- 최종 TitLeNet 관련 실험은 `small_head`를 사용했고, 이 설정에서 최종 validation/test 결과가 안정적이었다.

## 재현 명령어

최종 TitLeNet 학습:

```bash
/home/cvlab/anaconda3/envs/gemma_cuda/bin/python experiments/title_color_recommendation/run_full_training.py \
  --config configs/title_color_recommendation/full_training.yaml \
  --model-name titlenet \
  --epochs 20 \
  --learning-rate 0.0005 \
  --weight-decay 0.0001 \
  --batch-size 64 \
  --dropout 0.2 \
  --weight-init small_head \
  --activation gelu \
  --num-workers 4 \
  --device cuda \
  --seed 42 \
  --best-metric val_ndcg@5 \
  --scheduler cosine \
  --checkpoint-dir outputs/checkpoints/titlenet_ndcg3_eval \
  --log-path outputs/logs/titlenet_ndcg3_eval.jsonl \
  --report-path outputs/reports/model_evaluation/ndcg3_runs/titlenet_ndcg3_eval_report.md \
  --loss-plot-path outputs/reports/model_evaluation/ndcg3_runs/titlenet_ndcg3_eval_loss_curve.png \
  --ndcg-plot-path outputs/reports/model_evaluation/ndcg3_runs/titlenet_ndcg3_eval_ndcg_curve.png \
  --color-plot-path outputs/reports/model_evaluation/ndcg3_runs/titlenet_ndcg3_eval_colors.png
```

## 산출물

| artifact | path |
| --- | --- |
| 상세 문서 | `docs/title_color_recommendation/titlenet_experiment_summary.md` |
| TitLeNet 하이퍼파라미터 CSV | `outputs/reports/titlenet_hyperparameter_summary.csv` |
| TitLeNet 초기화 CSV | `outputs/reports/titlenet_initialization_summary.csv` |
| 짧은 요약 리포트 | `outputs/reports/titlenet_experiment_summary.md` |
| 최종 checkpoint | `outputs/checkpoints/titlenet_ndcg3_eval/checkpoint_best.pt` |
| 최종 log | `outputs/logs/titlenet_ndcg3_eval.jsonl` |
