# NDCG@3 Model Evaluation and Latency Comparison

## Runs

All listed models were evaluated from saved best checkpoints.

```text
titlenet
resnet18
resnet34
vit_tiny
convnext_tiny
efficientnet_b0
flatten_mlp
swin_tiny
```

## Test Metrics With Latency

| rank | model | test_loss | test_ndcg@3 | test_ndcg@5 | max_color_share | params | latency_b1_ms | latency_b64_ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `titlenet` | 0.006222 | 0.988514 | 0.990352 | 0.381842 | 0.184M | 1.831 | 2.936 |
| 2 | `resnet18` | 0.006409 | 0.987618 | 0.989619 | 0.389097 | 11.319M | 2.096 | 2.677 |
| 3 | `resnet34` | 0.006459 | 0.986784 | 0.988668 | 0.389316 | 21.427M | 3.788 | 4.827 |
| 4 | `vit_tiny` | 0.007884 | 0.985953 | 0.988576 | 0.390196 | 1.249M | 1.026 | 4.826 |
| 5 | `convnext_tiny` | 0.008993 | 0.982382 | 0.984856 | 0.398769 | 28.027M | 4.060 | 6.393 |
| 6 | `efficientnet_b0` | 0.010598 | 0.980445 | 0.983091 | 0.407562 | 4.344M | 6.732 | 6.863 |
| 7 | `flatten_mlp` | 0.058536 | 0.945153 | 0.952689 | 0.448230 | 10.167M | 0.110 | 0.160 |
| 8 | `swin_tiny` | 0.090539 | 0.927936 | 0.935918 | 0.462739 | 27.726M | 9.223 | 21.007 |

## Latency Ranking

### Batch 1

| rank | model | latency_b1_ms | test_ndcg@3 |
| ---: | --- | ---: | ---: |
| 1 | `flatten_mlp` | 0.110 | 0.945153 |
| 2 | `vit_tiny` | 1.026 | 0.985953 |
| 3 | `titlenet` | 1.831 | 0.988514 |
| 4 | `resnet18` | 2.096 | 0.987618 |
| 5 | `resnet34` | 3.788 | 0.986784 |
| 6 | `convnext_tiny` | 4.060 | 0.982382 |
| 7 | `efficientnet_b0` | 6.732 | 0.980445 |
| 8 | `swin_tiny` | 9.223 | 0.927936 |

### Batch 64

| rank | model | latency_b64_ms | ms_per_image | test_ndcg@3 |
| ---: | --- | ---: | ---: | ---: |
| 1 | `flatten_mlp` | 0.160 | 0.0025 | 0.945153 |
| 2 | `resnet18` | 2.677 | 0.0418 | 0.987618 |
| 3 | `titlenet` | 2.936 | 0.0459 | 0.988514 |
| 4 | `vit_tiny` | 4.826 | 0.0754 | 0.985953 |
| 5 | `resnet34` | 4.827 | 0.0754 | 0.986784 |
| 6 | `convnext_tiny` | 6.393 | 0.0999 | 0.982382 |
| 7 | `efficientnet_b0` | 6.863 | 0.1072 | 0.980445 |
| 8 | `swin_tiny` | 21.007 | 0.3282 | 0.927936 |

## Interpretation

- `titlenet` is the best model by `test_ndcg@3` in this checkpoint evaluation.
- `flatten_mlp` is the fastest model at batch 1 among the evaluated checkpoints.
- `max_color_share` is still useful for detecting collapse toward one dominant color.

## Recommendation

```text
Use the top-ranked quality model when recommendation quality is primary.
Use the batch-1 latency winner only if single-image response time is primary.
Reject models with low NDCG or high max_color_share for final service use.
```

## Artifacts

| artifact | path |
| --- | --- |
| checkpoint evaluation CSV | `outputs/reports/model_evaluation/checkpoint_eval_results.csv` |
| latency benchmark CSV | `outputs/reports/model_evaluation/latency/existing_models_latency_ndcg3_eval.csv` |
| `titlenet` checkpoint | `outputs/checkpoints/titlenet_ndcg3_eval/checkpoint_best.pt` |
| `resnet18` checkpoint | `outputs/checkpoints/resnet18_ndcg3_eval/checkpoint_best.pt` |
| `resnet34` checkpoint | `outputs/checkpoints/resnet34_ndcg3_eval/checkpoint_best.pt` |
| `vit_tiny` checkpoint | `outputs/checkpoints/vit_tiny_ndcg3_eval/checkpoint_best.pt` |
| `convnext_tiny` checkpoint | `outputs/checkpoints/convnext_tiny_ndcg3_eval/checkpoint_best.pt` |
| `efficientnet_b0` checkpoint | `outputs/checkpoints/efficientnet_b0_ndcg3_eval/checkpoint_best.pt` |
| `flatten_mlp` checkpoint | `outputs/checkpoints/flatten_mlp_ndcg3_eval/checkpoint_best.pt` |
| `swin_tiny` checkpoint | `outputs/checkpoints/swin_tiny_ndcg3_eval/checkpoint_best.pt` |
