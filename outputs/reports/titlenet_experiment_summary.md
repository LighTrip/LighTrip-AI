# TitLeNet Hyperparameter and Initialization Summary

상세 문서:

```text
docs/title_color_recommendation/titlenet_experiment_summary.md
```

## Scope

최종 TitLeNet과 직접 연결되는 항목만 유지한다.

```text
titlenet
simple_cnn_m_res_se
```

## Final Hyperparameters

| item | value |
| --- | --- |
| model_name | `titlenet` |
| learning_rate | `0.0005` |
| weight_decay | `0.0001` |
| dropout | `0.2` |
| activation | `gelu` |
| weight_init | `small_head` |
| batch_size | `64` |
| scheduler | `cosine` |
| best_metric | `val_ndcg@5` |

## Key Result

| run | best_epoch | best_val_loss | best_val_ndcg@5 | test_loss | test_ndcg@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `titlenet_ndcg3_eval` | 11 | 0.006219 | 0.990426 | 0.006222 | 0.990352 |

## Related Tables

| artifact | path |
| --- | --- |
| TitLeNet hyperparameter table | `outputs/reports/titlenet_hyperparameter_summary.csv` |
| TitLeNet initialization table | `outputs/reports/titlenet_initialization_summary.csv` |
