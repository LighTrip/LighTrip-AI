# TitLeNet Student QAT FP16 React Native Handoff

## 목적

TitLeNet Student QAT FP16 모델을 React Native 온디바이스 추론에 연동하기 위해 앱에서 필요한 산출물, 입력/출력 스펙, 검증 항목을 정리한다.

이 문서는 프론트 연동 전 전달용 문서이며, 실제 앱 런타임에서의 성능 측정과 호환성 검증은 별도 기기에서 수행한다.

## 배포 후보

| 항목 | 값 |
| --- | --- |
| model_id | `titlenet_student_qat_fp16` |
| source_model | `titlenet_student_qat_kd90` |
| precision | `fp16_internal_float32_io` |
| 최종 출력 | top-1 palette index 1개 |
| 배포 우선순위 | 1순위 후보 |

선정 기준:

- 실제 전처리 샘플 100개 기준 top-1 일치율이 `100%`이다.
- QAT FP16 모델의 NDCG@5 drop이 `-0.000136`으로 매우 작다.
- Python ONNX Runtime CPU batch1 기준 latency가 약 `0.737613 ms`이다.
- Static INT8 후보는 현재 top-1 일치율 기준을 만족하지 못해 배포 후보에서 제외한다.

## 전달 파일

| 구분 | 경로 | 앱 사용 여부 |
| --- | --- | --- |
| top-1 ONNX | `outputs/title_color_recommendation/deployment/titlenet_student_qat_fp16_top1.onnx` | 필수 |
| palette | `outputs/title_color_recommendation/deployment/palette.json` | 필수 |
| logits ONNX | `outputs/title_color_recommendation/deployment/titlenet_student_qat_fp16_logits.onnx` | 디버그/검증용 |
| metadata | `outputs/title_color_recommendation/deployment/titlenet_student_qat_fp16_metadata.json` | 연동 확인용 |

앱의 최종 색상 추천에는 `titlenet_student_qat_fp16_top1.onnx`만 사용한다. `logits` 모델은 배포 전 디버깅, Python 결과 비교, confidence 분석이 필요할 때만 사용한다.

## 입력 스펙

모델 입력은 전처리된 제목 영역 RGB와 텍스트 mask를 결합한 4채널 tensor이다.

| 항목 | 값 |
| --- | --- |
| input name | `input` |
| shape | `[1, 4, 36, 136]` |
| layout | `NCHW` |
| dtype | `float32` |
| batch size | `1` |
| channel order | `R, G, B, mask` |
| RGB range | `0.0`-`1.0` |
| mask range | `0.0` 또는 `1.0` |

채널별 의미:

| channel index | 의미 | 값 범위 |
| ---: | --- | --- |
| 0 | red | `0.0`-`1.0` |
| 1 | green | `0.0`-`1.0` |
| 2 | blue | `0.0`-`1.0` |
| 3 | text mask | background `0.0`, text area `1.0` |

별도 mean/std normalization은 적용하지 않는다. RGB는 `uint8` 값을 `255.0`으로 나눈 float 값으로 넣고, mask는 binary float으로 넣는다.

## ROI 기준

학습과 검증에서 사용한 기준 입력 캔버스는 `150x200`이다.

| 항목 | 값 |
| --- | ---: |
| input width | `150` |
| input height | `200` |
| ROI box | `(7, 36, 143, 72)` |
| ROI width | `136` |
| ROI height | `36` |

픽셀 좌표는 다음 상대 좌표를 기준으로 계산한다.

| 항목 | 값 |
| --- | ---: |
| x1 | `0.05` |
| y1 | `0.18` |
| x2 | `0.95` |
| y2 | `0.36` |

계산 방식은 `x1=floor(width*x1)`, `y1=floor(height*y1)`, `x2=ceil(width*x2)`, `y2=ceil(height*y2)` 기준이다.

## 출력 스펙

앱 최종 배포 모델은 top-1 index만 반환한다.

| 항목 | 값 |
| --- | --- |
| model file | `titlenet_student_qat_fp16_top1.onnx` |
| output name | `top1_index` |
| shape | `[1]` |
| dtype | `int64` |
| 값 범위 | `0`-`31` |
| 의미 | `palette.json`의 color id |

출력값은 palette id로 사용한다.

```text
top1_index = model(input)[0]
selected_color = palette[top1_index]
```

`top1_index`는 `0..31` 범위의 정수여야 한다. React Native 런타임에서 `int64` output을 어떤 배열 타입으로 반환하는지 확인하고, 앱 내부 palette lookup에는 number 타입으로 변환해서 사용한다.

## Palette 매핑

앱은 `outputs/title_color_recommendation/deployment/palette.json`을 모델과 함께 번들링한다.

매핑 기준:

| 항목 | 값 |
| --- | --- |
| palette count | `32` |
| index field | `id` |
| valid ids | `0..31` |
| model output mapping | `top1_index == palette item.id` |

최종 앱 결과는 최소 다음 값을 만들 수 있어야 한다.

```json
{
  "color_index": 0,
  "hex": "#FFFFFF"
}
```

## 앱 구현 요청 사항

React Native 연동 시 다음 항목을 확인한다.

| 구분 | 요청 내용 |
| --- | --- |
| asset bundle | `top1.onnx`와 `palette.json`을 앱 asset에 포함한다. |
| runtime | 선택한 ONNX Runtime이 FP16 내부 연산과 `int64` 출력을 지원하는지 확인한다. |
| preprocessing | ROI crop, RGB scaling, mask 생성, 4채널 NCHW tensor 생성을 앱에서 수행한다. |
| inference | 최종 추천은 `titlenet_student_qat_fp16_top1.onnx`만 호출한다. |
| postprocess | `top1_index`로 `palette.json`의 `id`를 조회한다. |
| error handling | output이 `0..31` 범위를 벗어나면 fallback 색상 또는 기본 색상을 사용한다. |

전처리는 현재 모델 내부에 포함되어 있지 않다. 따라서 앱에서 기존 학습/검증과 동일한 방식으로 ROI와 mask를 만들어야 한다.

## 앱 검증 체크리스트

연동 완료 후 다음 항목을 확인한다.

| 항목 | 기대값 |
| --- | --- |
| 모델 로드 | 앱 시작 또는 추론 직전 ONNX 로드 성공 |
| palette 로드 | `32`개 색상 로드 성공 |
| 입력 tensor shape | `[1, 4, 36, 136]` |
| 입력 dtype | `float32` |
| 출력 shape | `[1]` |
| 출력 dtype | `int64` 또는 런타임의 int64 대응 타입 |
| 출력 범위 | `0..31` |
| palette lookup | `top1_index`와 동일한 `id` 조회 성공 |
| sample smoke test | 동일 입력에서 반복 실행 결과가 동일 |

## 기기 성능 측정 기준

Python ONNX Runtime CPU latency는 참고값일 뿐이며, 앱에서는 실제 타깃 기기 기준으로 측정한다.

권장 측정 항목:

| 항목 | 설명 |
| --- | --- |
| cold start latency | 모델 첫 로드와 첫 추론 시간 |
| warm inference latency | 모델 로드 후 반복 추론 시간 |
| P50/P95 latency | 100회 이상 반복 기준 |
| memory usage | 모델 로드 전후 메모리 변화 |
| crash/error | Android/iOS 각각 런타임 에러 여부 |
| device info | OS, 기기명, 칩셋, 앱 빌드 타입 |

현재 Python 기준 참고값:

| 항목 | 값 |
| --- | ---: |
| top1 model size | `0.153341 MB` |
| Python ONNX Runtime CPU batch1 | `0.737613 ms` |

## 참조 문서

- `docs/title_color_recommendation/titlenet_mobile_inference_spec.md`
- `docs/title_color_recommendation/titlenet_student_quantization_prep.md`
- `outputs/reports/model_evaluation/onnx/titlenet_student_qat_fp16_deployment_report.md`
- `outputs/title_color_recommendation/deployment/titlenet_student_qat_fp16_metadata.json`
