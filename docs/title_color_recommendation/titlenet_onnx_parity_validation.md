# TitLeNet PyTorch-ONNX Parity Validation

## 목적

TitLeNet ONNX export 이후 실제 전처리된 ROI/mask 샘플 기준으로 PyTorch 모델과 ONNX 모델의 추론 결과가 동일하게 유지되는지 검증한다.

## 검증 대상

| 항목 | 경로 |
| --- | --- |
| PyTorch checkpoint | `outputs/checkpoints/titlenet_ndcg3_eval/checkpoint_best.pt` |
| logits ONNX | `outputs/title_color_recommendation/onnx/titlenet_logits.onnx` |
| top-1 ONNX | `outputs/title_color_recommendation/onnx/titlenet_top1.onnx` |
| palette | `data/title_color_recommendation/processed/palette.json` |

## 실행 명령

```bash
/home/cvlab/anaconda3/envs/gemma_cuda/bin/python \
  experiments/title_color_recommendation/validate_titlenet_onnx.py
```

기본 설정:

| 항목 | 값 |
| --- | --- |
| split | `test` |
| sample_count | `100` |
| seed | `42` |
| max_abs_diff threshold | `1e-4` |
| mean_abs_diff threshold | `1e-5` |

샘플은 보안용 난수나 PRNG를 사용하지 않고, `seed:index`의 SHA-256 digest 기준으로 결정적으로 선택한다.

## 산출물

```text
outputs/reports/model_evaluation/onnx/titlenet_onnx_parity_report.md
outputs/reports/model_evaluation/onnx/titlenet_onnx_parity_metrics.json
```

산출물은 `outputs/` 하위 파일이므로 Git에는 포함하지 않는다.

## 검증 항목

- 실제 전처리된 입력 tensor shape가 `[1, 4, 36, 136]`인지 확인한다.
- `titlenet_logits.onnx` 출력 shape가 `[1, 32]`인지 확인한다.
- `titlenet_top1.onnx` 출력 shape가 `[1]`, dtype이 `int64`인지 확인한다.
- PyTorch logits와 ONNX logits의 `max_abs_diff`, `mean_abs_diff`를 계산한다.
- PyTorch top-1, ONNX logits argmax, ONNX top-1 모델 출력이 모두 일치하는지 확인한다.
- 참고 지표로 top-3/top-5 ordered match를 계산한다.
- top-1 index가 `0..31` 범위의 palette id와 매핑되는지 확인한다.

## 기준 결과

`test` split에서 고정 seed `42`로 100개 샘플을 검증한 결과:

| metric | value | threshold |
| --- | ---: | ---: |
| top1_agreement | `100.00%` | `100.00%` |
| top3_agreement | `100.00%` | - |
| top5_agreement | `100.00%` | - |
| max_abs_diff | `3.3378601e-06` | `1e-4` |
| mean_abs_diff | `2.4808156e-07` | `1e-5` |

결과: pass.
