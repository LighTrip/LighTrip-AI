# Places365 확장 데이터셋 품질 검토

검토 대상은 `data/category_classifier/places365_v2`와 `configs/places365_categories_v2.json`입니다. 기존 예시 label 중 v2에서 제외된 `cafeteria`, `bookstore` 등은 `data/category_classifier/places365_v1`에 남아 있는 샘플을 함께 확인했습니다.

## 요약

- 총 샘플 수는 7,000장입니다.
- 카테고리별 샘플 수는 7개 카테고리 모두 1,000장으로 균형이 맞습니다.
- split도 카테고리별로 `train 800`, `valid 100`, `test 100`으로 균형이 맞습니다.
- 카테고리 단위 imbalance는 없지만, Places365 label 단위 imbalance는 있습니다.
- `카페`는 현재 `coffee_shop` 1개 label만으로 1,000장을 구성하므로 카페 내부 다양성이 가장 낮습니다.
- v2 설정에 포함된 56개 Places365 label 중 실제 수집 샘플이 있는 label은 36개이고, 20개는 현재 0장입니다.
- 서비스 카테고리 기준과 충돌 가능성이 큰 label은 label 전체 제외 또는 sample-level filtering 대상으로 분리하는 것이 좋습니다.

## 카테고리별 분포

| category | samples | train | valid | test | positive Places365 labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| 카페 | 1,000 | 800 | 100 | 100 | 1 |
| 식당 | 1,000 | 800 | 100 | 100 | 5 |
| 술집 | 1,000 | 800 | 100 | 100 | 3 |
| 문화 | 1,000 | 800 | 100 | 100 | 4 |
| 쇼핑 | 1,000 | 800 | 100 | 100 | 7 |
| 운동 | 1,000 | 800 | 100 | 100 | 8 |
| 공원 | 1,000 | 800 | 100 | 100 | 8 |

판단: 카테고리별 양은 균형적입니다. 다만 카테고리 내부 label 다양성이 다르므로, 모델이 `카페=coffee shop`처럼 좁은 시각 단서에 과적합할 위험이 있습니다.

## Places365 Label별 분포

| category | Places365 label | samples | 검토 결과 |
| --- | --- | ---: | --- |
| 카페 | coffee_shop | 1,000 | 유지. 카페 대표성이 가장 높음 |
| 식당 | fastfood_restaurant | 200 | 유지 |
| 식당 | food_court | 200 | 유지하되 쇼핑몰/행사장/대기 공간처럼 보이는 샘플 필터링 필요 |
| 식당 | pizzeria | 200 | 유지 |
| 식당 | restaurant | 200 | 유지 |
| 식당 | restaurant_patio | 200 | 유지하되 풍경/빈 테라스 위주 샘플 필터링 필요 |
| 술집 | bar | 334 | 유지 |
| 술집 | beer_garden | 334 | 유지하되 공원/야외 행사처럼 보이는 샘플 필터링 필요 |
| 술집 | beer_hall | 332 | 유지 |
| 문화 | amphitheater | 250 | 유지하되 공원/건축물 계단처럼 보이는 샘플 필터링 필요 |
| 문화 | art_gallery | 250 | 유지 |
| 문화 | natural_history_museum | 250 | 유지 |
| 문화 | science_museum | 250 | 유지 |
| 쇼핑 | clothing_store | 143 | 유지 |
| 쇼핑 | department_store | 143 | 유지 |
| 쇼핑 | gift_shop | 143 | 유지 |
| 쇼핑 | jewelry_shop | 142 | 유지 |
| 쇼핑 | shoe_shop | 143 | 유지 |
| 쇼핑 | supermarket | 143 | 유지 |
| 쇼핑 | toyshop | 143 | 유지 |
| 운동 | baseball_field | 125 | 유지 |
| 운동 | bowling_alley | 125 | 유지 |
| 운동 | boxing_ring | 125 | 유지 |
| 운동 | football_field | 125 | 유지 |
| 운동 | golf_course | 125 | 유지하되 공원/풍경 단서가 강한 샘플 주의 |
| 운동 | martial_arts_gym | 125 | 유지 |
| 운동 | ski_slope | 125 | 유지하되 여행/자연 풍경 단서 주의 |
| 운동 | soccer_field | 125 | 유지 |
| 공원 | botanical_garden | 125 | 유지. 서비스 기준상 공원으로 사용 |
| 공원 | formal_garden | 125 | 유지 |
| 공원 | japanese_garden | 125 | 유지 |
| 공원 | park | 125 | 유지 |
| 공원 | picnic_area | 125 | 유지. 서비스 기준상 공원으로 사용 |
| 공원 | playground | 125 | 유지. 운동보다 공원/놀이 공간 기준에 가까움 |
| 공원 | topiary_garden | 125 | 유지 |
| 공원 | zen_garden | 125 | 유지 |

## 0장 수집 Label

아래 label은 v2 설정에 포함되어 있지만 `metadata.csv` 기준 실제 수집 샘플이 없습니다. 현재 학습 데이터에는 영향을 주지 않지만, 이후 재수집 시 품질 검토 대상입니다.

| category | Places365 label | 권장 처리 |
| --- | --- | --- |
| 식당 | diner_outdoor | 샘플 확보 시 식당 유지 가능 |
| 술집 | pub_indoor | 샘플 확보 시 술집 유지 가능 |
| 문화 | library_indoor | 샘플 확보 시 문화 유지 가능 |
| 문화 | movie_theater_indoor | 샘플 확보 시 문화 유지 가능 |
| 문화 | museum_indoor | 샘플 확보 시 문화 유지 가능 |
| 쇼핑 | bazaar_indoor | 샘플 확보 시 쇼핑 유지 가능 |
| 쇼핑 | bazaar_outdoor | 야시장/관광지 경계 검토 필요 |
| 쇼핑 | flea_market_indoor | 샘플 확보 시 쇼핑 유지 가능 |
| 쇼핑 | general_store_indoor | 샘플 확보 시 쇼핑 유지 가능 |
| 쇼핑 | market_indoor | 샘플 확보 시 쇼핑 유지 가능 |
| 쇼핑 | market_outdoor | 시장/거리/관광지 경계 검토 필요 |
| 쇼핑 | shopping_mall_indoor | 샘플 확보 시 쇼핑 유지 가능 |
| 운동 | athletic_field_outdoor | 샘플 확보 시 운동 유지 가능 |
| 운동 | basketball_court_indoor | 샘플 확보 시 운동 유지 가능 |
| 운동 | gymnasium_indoor | 샘플 확보 시 운동 유지 가능 |
| 운동 | ice_skating_rink_indoor | 샘플 확보 시 운동 유지 가능 |
| 운동 | ice_skating_rink_outdoor | 샘플 확보 시 운동 유지 가능 |
| 운동 | swimming_pool_indoor | 샘플 확보 시 운동 유지 가능 |
| 운동 | swimming_pool_outdoor | 샘플 확보 시 운동 유지 가능 |
| 운동 | volleyball_court_outdoor | 샘플 확보 시 운동 유지 가능 |

## Mapping 검토 결론

현재 v2의 큰 방향인 `direct_only_high_confidence`는 서비스 관점에서 적절합니다. v1의 넓은 매핑보다 noisy label을 줄이는 데 유리합니다.

수정/보완해야 할 점은 세 가지입니다.

1. `bookstore`는 v2에서 빠져 있지만 제외 목록에 명시되어 있지 않습니다. 실제 샘플은 서점 판매 공간과 도서관/열람 공간 느낌이 섞여 있어 `문화`와 `쇼핑` 양쪽으로 흔들립니다. v2 기준에서는 `exclude`로 명시하는 것이 안전합니다.
2. `cafeteria`, `ice_cream_parlor`, `sushi_bar`, `discotheque`, `wet_bar`, `lawn`, `auditorium` 등은 기존 샘플 확인 결과 제외 판단이 타당합니다.
3. `food_court`, `restaurant_patio`, `beer_garden`, `amphitheater`, `golf_course`는 label 전체를 제거하기보다 sample-level filtering 대상으로 관리하는 것이 좋습니다. `botanical_garden`, `picnic_area`는 서비스 기준상 공원으로 바로 포함합니다.

## Category Imbalance 판단

카테고리별 sample imbalance는 없습니다. 모든 카테고리가 1,000장으로 동일합니다.

다만 label-level imbalance는 존재합니다.

- `카페`: `coffee_shop`만 존재해 카테고리 다양성이 부족합니다.
- `술집`: `bar`, `beer_garden`, `beer_hall` 3개 label에 집중됩니다.
- `문화`: 실제 positive label은 4개뿐이며 `library_indoor`, `movie_theater_indoor`, `museum_indoor`는 0장입니다.
- `식당`: `diner_outdoor`는 0장이고 실제로는 5개 label만 사용됩니다.
- `쇼핑`, `운동`, `공원`: 카테고리 내부 분산은 비교적 좋지만, 야외/관광/풍경 단서가 섞이는 label이 있습니다.

## 권장 후속 처리

- 학습에는 `configs/places365_category_mapping_v2.json`의 `keep`과 `keep_sample_review`를 사용합니다.
- `exclude` label은 데이터 수를 늘리기 위해 재투입하지 않습니다.
- `keep_sample_review` label은 샘플 단위로 1차 필터링한 뒤 학습에 포함합니다.
- `기타` 카테고리는 `data/category_classifier/places365_v2`에 없습니다. 서비스 분류기가 `기타`를 필요로 한다면 별도 negative/unknown 데이터셋으로 보강해야 합니다.
