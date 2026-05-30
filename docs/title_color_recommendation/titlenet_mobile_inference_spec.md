# TitLeNet On-Device Inference Spec

## 목적

TitLeNet을 온디바이스 추론용 모델로 배포하기 위해 입력, 출력, 후처리 스펙을 고정한다. 이 문서는 ONNX export, 변환 검증, 양자화 실험에서 공통 기준으로 사용한다.

## 범위

- 대상 모델: `titlenet`
- 대상 태스크: 제목 색상 추천
- 최종 추천 방식: top-1 색상 1개 선택
- 모델 출력 후보 수: 32개 고정 palette 색상
- 기준 config: `configs/title_color_recommendation/default.yaml`
- 참조용 배포 config: `configs/title_color_recommendation/titlenet_mobile_inference.json`

프론트 연동 방식, 네이티브 런타임 선택, 앱 UI 처리는 이 문서의 범위에 포함하지 않는다.

## 입력 스펙

TitLeNet 모델 입력은 전처리된 title ROI RGB와 text mask를 결합한 4채널 tensor이다.

| 항목 | 값 |
| --- | --- |
| shape | `[1, 4, 36, 136]` |
| layout | `NCHW` |
| dtype | `float32` |
| batch size | `1` 고정 |
| channel order | `R, G, B, mask` |
| RGB range | `0.0`-`1.0` |
| mask range | `0.0` 또는 `1.0` |

채널 의미:

| channel index | 의미 | 값 범위 |
| ---: | --- | --- |
| 0 | red | `0.0`-`1.0` |
| 1 | green | `0.0`-`1.0` |
| 2 | blue | `0.0`-`1.0` |
| 3 | text mask | background `0.0`, text area `1.0` |

별도 mean/std normalization은 적용하지 않는다. RGB는 `uint8` 이미지 값을 `255.0`으로 나눈 값이며, mask는 threshold 후 binary float으로 사용한다.

## ROI 기준

학습 파이프라인의 원본 기준 입력 캔버스는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| input width | `150` |
| input height | `200` |

상대 ROI 기준:

| 항목 | 값 |
| --- | ---: |
| x1 | `0.05` |
| y1 | `0.18` |
| x2 | `0.95` |
| y2 | `0.36` |

픽셀 ROI 기준:

| 항목 | 값 |
| --- | ---: |
| x | `7` |
| y | `36` |
| width | `136` |
| height | `36` |
| box | `(7, 36, 143, 72)` |

픽셀 좌표는 학습 코드와 동일하게 `x1=floor(width*x1)`, `y1=floor(height*y1)`, `x2=ceil(width*x2)`, `y2=ceil(height*y2)` 방식으로 계산한다.

## 출력 스펙

모델 출력은 32개 palette 색상에 대한 raw score/logit이다.

| 항목 | 값 |
| --- | --- |
| shape | `[1, 32]` |
| dtype | `float32` |
| 의미 | palette color별 score/logit |
| 후처리 | `argmax` |
| 최종 반환 | top-1 color index |

후처리는 다음 기준으로 고정한다.

```text
logits = model(input)
top1_index = argmax(logits[0])
```

top-1 선택에는 softmax가 필요하지 않다. confidence 표시가 필요한 실험에서는 softmax 확률을 추가 계산할 수 있지만, 최종 색상 선택 기준은 항상 raw logits의 `argmax`이다.

동점이 발생하면 가장 작은 index를 선택한다.

## Palette 매핑

Palette 파일은 다음 경로를 기준으로 한다.

```text
data/title_color_recommendation/processed/palette.json
```

`top1_index`는 palette entry의 `id`와 매핑한다.

예시:

```json
{
  "id": 0,
  "hex": "#FFFFFF",
  "name": "pure_white"
}
```

최종 색상 결과는 최소 다음 필드를 가진다.

```json
{
  "color_index": 0,
  "hex": "#FFFFFF"
}
```

## 배포용 Config 항목

온디바이스 export와 검증에서 사용할 config는 다음 항목을 포함한다.

| 항목 | 설명 |
| --- | --- |
| `schema_version` | 모바일 추론 스펙 버전 |
| `model.name` | 모델 이름 |
| `model.task` | 수행 태스크 |
| `model.format` | export 대상 포맷 |
| `input_size` | ROI 추출 전 기준 이미지 크기 |
| `roi.relative` | 상대 ROI 좌표 |
| `roi.pixel` | 고정 픽셀 ROI 좌표 |
| `model_input` | 입력 shape, dtype, layout, channel order |
| `model_output` | 출력 shape, dtype, 후처리 |
| `palette` | palette 파일 경로와 index 매핑 기준 |

참조 파일:

```text
configs/title_color_recommendation/titlenet_mobile_inference.json
```

## 검증 기준

후속 ONNX export와 검증 작업에서는 이 스펙을 기준으로 다음을 확인한다.

- 입력 tensor shape가 `[1, 4, 36, 136]`인지 확인한다.
- 출력 tensor shape가 `[1, 32]`인지 확인한다.
- PyTorch와 변환 모델의 `top1_index`가 일치하는지 확인한다.
- `top1_index`가 `palette.json`의 `id`와 올바르게 매핑되는지 확인한다.
