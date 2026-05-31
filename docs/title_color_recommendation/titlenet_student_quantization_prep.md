# TitLeNet Student Quantization Prep

## 목적

최고 Student 후보인 `titlenet_student + warm_start kd_90_10` checkpoint를 양자화 전 FP32 기준 모델로 고정한다. 이 단계에서는 아직 INT8 양자화를 수행하지 않고, 이후 양자화 결과와 비교할 ONNX, parity, baseline, calibration 산출물을 만든다.

## 기준 모델

| item | value |
| --- | --- |
| model | `titlenet_student` |
| training | `warm_start kd_90_10` |
| checkpoint | `outputs/checkpoints/titlenet_student_kd_weight_sweep/warm_start/kd_90_10/checkpoint_best.pt` |
| loss | `0.9 * soft_label_loss + 0.1 * teacher_KD_loss` |
| temperature | `2.0` |
| input | `[1, 4, 36, 136]` float32 |
| output logits | `[1, 32]` float32 |
| output top-1 | `[1]` int64 palette index |

## 산출물

```text
outputs/title_color_recommendation/onnx/titlenet_student_warm_kd90_logits.onnx
outputs/title_color_recommendation/onnx/titlenet_student_warm_kd90_top1.onnx
outputs/title_color_recommendation/onnx/titlenet_student_warm_kd90_onnx_export_summary.json

outputs/reports/model_evaluation/onnx/titlenet_student_warm_kd90_parity_report.md
outputs/reports/model_evaluation/onnx/titlenet_student_warm_kd90_parity_metrics.json

outputs/reports/model_evaluation/onnx/titlenet_student_warm_kd90_baseline_report.md
outputs/reports/model_evaluation/onnx/titlenet_student_warm_kd90_baseline_metrics.json

outputs/title_color_recommendation/quantization/calibration_samples/titlenet_student_warm_kd90/
outputs/title_color_recommendation/quantization/titlenet_student_warm_kd90_calibration_manifest.json
```

## 실행 명령

```bash
/home/cvlab/anaconda3/envs/gemma_cuda/bin/python \
  experiments/title_color_recommendation/prepare_titlenet_student_quantization_baseline.py
```

기본 실행은 다음을 모두 수행한다.

- Student FP32 logits ONNX export
- Student FP32 top-1 ONNX export
- PyTorch Student vs ONNX Student parity 검증
- FP32 baseline 리포트 생성
- INT8 static quantization용 calibration 샘플 저장

## 빠른 확인용 실행

개발 중에는 샘플 수와 benchmark 횟수를 줄여 smoke test를 수행할 수 있다.

```bash
/home/cvlab/anaconda3/envs/gemma_cuda/bin/python \
  experiments/title_color_recommendation/prepare_titlenet_student_quantization_baseline.py \
  --parity-sample-count 5 \
  --calibration-sample-count 5 \
  --latency-warmup-steps 1 \
  --latency-benchmark-steps 2
```

## 검증 기준

```text
top1_agreement = 100%
max_abs_diff <= 1e-4
mean_abs_diff <= 1e-5
top1 output dtype = int64
top1 index range = 0..31
```

## 양자화 단계에서 사용할 기준

양자화 후에는 이 작업에서 생성한 FP32 baseline과 다음 항목을 비교한다.

| 비교 항목 | FP32 기준 |
| --- | --- |
| 정확도 | `titlenet_student_warm_kd90_baseline_metrics.json` |
| PyTorch-ONNX parity | `titlenet_student_warm_kd90_parity_metrics.json` |
| calibration 입력 | `titlenet_student_warm_kd90_calibration_manifest.json` |
| logits ONNX | `titlenet_student_warm_kd90_logits.onnx` |
| top-1 ONNX | `titlenet_student_warm_kd90_top1.onnx` |

## 주의

정확도 비교는 `logits` 모델 기준으로 수행한다. `top1` 모델은 내부에 `ArgMax`가 포함되어 있으므로, 최종 배포 출력 index 확인에 사용한다.
