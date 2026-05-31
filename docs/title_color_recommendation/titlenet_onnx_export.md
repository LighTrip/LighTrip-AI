# TitLeNet ONNX Export

## 목적

학습된 TitLeNet PyTorch checkpoint를 ONNX 모델로 변환한다. 온디바이스 배포에서는 최종 색상 1개만 사용하므로, 검증용 logits 모델과 배포용 top-1 모델을 함께 생성한다.

## Export 대상

| 모델 | 입력 | 출력 | 용도 |
| --- | --- | --- | --- |
| `titlenet_logits.onnx` | `[1, 4, 36, 136]` float32 | `[1, 32]` float32 logits | PyTorch-ONNX 검증용 |
| `titlenet_top1.onnx` | `[1, 4, 36, 136]` float32 | `[1]` int64 index | 온디바이스 배포용 |

`titlenet_top1.onnx`는 TitLeNet logits에 `argmax(logits, dim=1)` 후처리를 포함한다. 출력값은 `0..31` 범위의 palette index이며, `data/title_color_recommendation/processed/palette.json`의 `id`와 매핑한다.

## 기본 산출물

```text
outputs/title_color_recommendation/onnx/titlenet_logits.onnx
outputs/title_color_recommendation/onnx/titlenet_top1.onnx
outputs/title_color_recommendation/onnx/titlenet_onnx_export_summary.json
```

## 실행 명령

필요 패키지:

```bash
pip install onnx onnxruntime
```

```bash
/home/cvlab/anaconda3/envs/gemma_cuda/bin/python \
  scripts/title_color_recommendation/export_titlenet_onnx.py \
  --checkpoint outputs/checkpoints/titlenet_ndcg3_eval/checkpoint_best.pt \
  --output-dir outputs/title_color_recommendation/onnx \
  --opset 17
```

Student 최고 후보는 기존 TitLeNet 산출물과 구분하기 위해 파일명을 분리한다.

```bash
/home/cvlab/anaconda3/envs/gemma_cuda/bin/python \
  scripts/title_color_recommendation/export_titlenet_onnx.py \
  --checkpoint outputs/checkpoints/titlenet_student_kd_weight_sweep/warm_start/kd_90_10/checkpoint_best.pt \
  --logits-output outputs/title_color_recommendation/onnx/titlenet_student_warm_kd90_logits.onnx \
  --top1-output outputs/title_color_recommendation/onnx/titlenet_student_warm_kd90_top1.onnx \
  --summary-output outputs/title_color_recommendation/onnx/titlenet_student_warm_kd90_onnx_export_summary.json \
  --opset 17
```

ONNX graph check까지 수행하려면 실행 환경에 `onnx` 패키지가 필요하다. 또한 top-1 ONNX 모델의 실제 dummy inference 결과까지 확인하려면 `onnxruntime` 패키지가 필요하다.

`onnxruntime` 없이 export와 graph check만 수행하려면 다음 옵션을 사용할 수 있다.

```bash
/home/cvlab/anaconda3/envs/gemma_cuda/bin/python \
  scripts/title_color_recommendation/export_titlenet_onnx.py \
  --skip-onnxruntime-check
```

## 검증 항목

스크립트는 dummy input 기준으로 다음을 확인한다.

- PyTorch logits 출력 shape: `[1, 32]`
- PyTorch top-1 wrapper 출력 shape: `[1]`
- PyTorch top-1 wrapper 출력 dtype: `int64`
- top-1 index 범위: `0 <= index < 32`
- ONNX 파일 생성 여부와 파일 크기
- ONNX graph check와 output shape/dtype 확인
- `onnxruntime` 패키지가 설치된 경우 ONNX Runtime dummy inference 확인

## 참조 스펙

온디바이스 입력, ROI, palette 매핑 기준은 다음 문서와 config를 따른다.

```text
docs/title_color_recommendation/titlenet_mobile_inference_spec.md
configs/title_color_recommendation/titlenet_mobile_inference.json
```
