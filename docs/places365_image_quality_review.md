# Places365 v2 이미지 품질 검사 결과

검사 대상은 `data_places365_2/metadata.csv` 기준 7,000장입니다. 이 단계는 초안 생성이나 SVM 학습이 아니라, 초안 생성에 투입할 이미지 후보를 선별하기 위한 이미지 데이터 품질 검사입니다.

## 산출물

| file | rows | 용도 |
| --- | ---: | --- |
| `data_places365_2/quality/quality_manifest.jsonl` | 7,000 | 전체 이미지별 품질 검사 결과 |
| `data_places365_2/quality/draft_candidates.jsonl` | 5,763 | 초안 생성에 바로 투입 가능한 strict 후보 |
| `data_places365_2/quality/accepted_images.jsonl` | 5,763 | `draft_candidates.jsonl`과 동일한 accepted 이미지 |
| `data_places365_2/quality/review_required_images.jsonl` | 1,233 | 이미지 파일은 정상이나 label 경계상 수동 검토가 필요한 이미지 |
| `data_places365_2/quality/rejected_images.jsonl` | 4 | hard fail로 제외할 이미지 |
| `data_places365_2/quality/summary.json` | 1 | 전체 품질 검사 요약 |

## 검사 기준

Hard fail 기준:

- 이미지 파일이 없음
- PIL로 열 수 없음
- 한 변이 128px 미만
- 종횡비가 `0.33` 미만 또는 `3.0` 초과
- 정확히 같은 파일 hash가 중복됨
- mapping decision이 `keep` 또는 `keep_sample_review`가 아님

Manual review 기준:

- mapping decision이 `keep_sample_review`
- RGB/L 외 mode
- 매우 어둡거나 밝은 이미지
- 대비가 매우 낮은 이미지
- perceptual hash가 같은 유사 이미지

이번 검사에서는 hard fail은 모두 `extreme_aspect_ratio`였습니다.

## 전체 결과

| status | images | 의미 |
| --- | ---: | --- |
| accepted | 5,763 | 초안 생성에 바로 사용 가능 |
| review_required | 1,233 | 파일 품질은 통과했지만 label 경계 검토 필요 |
| rejected | 4 | 초안 생성에서 제외 |

## Accepted 이미지 분포

| category | train | valid | test | total |
| --- | ---: | ---: | ---: | ---: |
| 카페 | 800 | 100 | 100 | 1,000 |
| 식당 | 466 | 70 | 62 | 598 |
| 술집 | 518 | 75 | 73 | 666 |
| 문화 | 602 | 75 | 73 | 750 |
| 쇼핑 | 800 | 100 | 100 | 1,000 |
| 운동 | 592 | 80 | 77 | 749 |
| 공원 | 800 | 100 | 100 | 1,000 |

`accepted`만 사용하면 데이터 품질은 가장 안전하지만 식당, 술집, 문화, 운동 쪽 샘플 수가 줄어듭니다. 이는 해당 카테고리의 일부 `keep_sample_review` label을 strict 후보에서 제외했기 때문입니다. 공원은 `botanical_garden`, `picnic_area`를 accepted로 포함해 1,000장을 유지합니다.

## Review Required Label

아래 label은 이미지 파일 자체는 정상이나 서비스 카테고리 경계상 수동 검토 후 포함하는 것이 좋습니다.

| Places365 label | category | images | 이유 |
| --- | --- | ---: | --- |
| food_court | 식당 | 200 | 쇼핑몰/공용 좌석/행사장 혼동 가능 |
| restaurant_patio | 식당 | 200 | 빈 테라스/풍경/리조트 혼동 가능 |
| beer_garden | 술집 | 334 | 야외 행사/공원 혼동 가능 |
| amphitheater | 문화 | 249 | 공원 시설/건축물 계단 혼동 가능 |
| golf_course | 운동 | 125 | 잔디/풍경 중심 샘플 혼동 가능 |
| ski_slope | 운동 | 125 | 자연/여행 풍경 혼동 가능 |

## Rejected 이미지

| id | path | reason |
| --- | --- | --- |
| culture_amphitheater_00009 | `data_places365_2/문화/amphitheater/00009.jpg` | extreme_aspect_ratio |
| restaurant_fastfood_restaurant_00063 | `data_places365_2/식당/fastfood_restaurant/00063.jpg` | extreme_aspect_ratio |
| restaurant_fastfood_restaurant_00123 | `data_places365_2/식당/fastfood_restaurant/00123.jpg` | extreme_aspect_ratio |
| exercise_soccer_field_00071 | `data_places365_2/운동/soccer_field/00071.jpg` | extreme_aspect_ratio |

## 최종 결론

초안 생성 데이터셋 구축에는 우선 `data_places365_2/quality/draft_candidates.jsonl`만 사용하는 것이 안전합니다. 이 경우 5,763장의 이미지로 시작할 수 있습니다.

`review_required_images.jsonl`의 1,233장은 버리는 데이터가 아니라, label 경계가 애매해서 수동 샘플 검토 후 추가 투입할 후보입니다. strict 후보로 초안 생성을 먼저 돌리고, 데이터 수가 부족한 카테고리는 review_required에서 승인된 이미지를 추가하는 흐름을 권장합니다.
