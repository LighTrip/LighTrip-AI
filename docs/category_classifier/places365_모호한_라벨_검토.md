# Ambiguous Places365 Label 검토

이 문서는 Places365 label과 LighTrip 서비스 카테고리 사이의 경계가 모호한 항목을 정리합니다. 샘플은 `data/category_classifier/places365_v2`를 우선 검토했고, v2에서 제외된 과거 label은 `data/category_classifier/places365_v1`의 기존 샘플을 확인했습니다.

## 판단 기준

| decision | 의미 |
| --- | --- |
| keep | 서비스 카테고리와 직접 대응되어 유지 |
| keep_sample_review | label은 유지하되 noisy sample 필터링 필요 |
| exclude | 카테고리 충돌이 커서 학습 데이터에서 제외 |
| zero_sample | 현재 설정에는 있으나 실제 수집 샘플 0장 |

## 우선 검토 Label

| Places365 label | 현재/후보 카테고리 | decision | 검토 내용 | 권장 처리 |
| --- | --- | --- | --- | --- |
| cafeteria | 식당 / 카페 | exclude | 식당, 구내식당, 행사 식사 공간, 일반 테이블 공간이 섞임. 카페 단서가 일관되지 않고 식당으로도 안정적이지 않음 | v2 제외 유지 |
| bookstore | 쇼핑 / 문화 | exclude | 책 판매 공간, 도서관형 서가, 열람/전시 느낌이 섞임. 서비스에서 서점을 쇼핑으로 볼지 문화로 볼지 기준이 흔들림 | v2 제외 목록에 명시 |
| beer_garden | 술집 / 공원 | keep_sample_review | 맥주/야외 음주 단서가 있는 샘플은 술집에 적합하지만 잔디, 야외 행사, 공원처럼 보이는 샘플도 있음 | 술집 유지, 음료/테이블/펍 단서 없는 야외 샘플 제외 |
| botanical_garden | 공원 / 문화 | keep | 서비스 기준상 식물원/정원 장면은 공원으로 사용하기로 결정 | 공원 유지 |
| food_court | 식당 / 쇼핑 | keep_sample_review | 푸드코트, 매장, 쇼핑몰 내부, 대기 공간이 섞임. 음식 섭취 공간 단서가 없으면 쇼핑/실내 공용 공간처럼 보임 | 식당 유지, 음식점/식사 좌석 단서 없는 샘플 제외 |
| restaurant_patio | 식당 / 카페 / 술집 / 여행 | keep_sample_review | 야외 테이블과 식당 테라스는 적합하지만 빈 테라스, 풍경, 리조트/전망 공간 샘플이 섞임 | 식당 유지, 식사 공간 단서 없는 풍경 샘플 제외 |
| amphitheater | 문화 / 공원 | keep_sample_review | 공연장/원형극장으로 볼 수 있지만 빈 야외 계단, 공원 시설, 관광지 건축물처럼 보이는 샘플이 많음 | 문화 유지, 무대/관람석 단서가 약한 샘플은 필터링 |
| picnic_area | 공원 / 식당 | keep | 서비스 기준상 피크닉 장소는 공원으로 사용하기로 결정 | 공원 유지 |
| playground | 공원 / 운동 | keep | 놀이 시설 중심으로 공원 카테고리에 적합. 운동보다는 야외 놀이/공원 시설 단서가 우세 | 공원 유지 |
| golf_course | 운동 / 공원 | keep_sample_review | 골프장 단서가 있으면 운동에 적합. 잔디/풍경만 보이는 샘플은 공원처럼 보일 수 있음 | 운동 유지, 골프 시설 단서 없는 풍경 샘플 제외 |
| ski_slope | 운동 / 자연 / 여행 | keep_sample_review | 스키장 경사면은 운동에 적합. 자연 설경/관광지처럼 보이는 샘플은 혼동 가능 | 운동 유지, 스키 활동/시설 단서 확인 |

## 제외 유지가 타당한 Label

| Places365 label | 후보 카테고리 | 검토 내용 | 권장 처리 |
| --- | --- | --- | --- |
| bakery_shop | 카페 / 쇼핑 | 카페보다는 제과 판매점 또는 매장 진열 이미지가 섞일 가능성이 높음 | exclude |
| delicatessen | 식당 / 쇼핑 | 음식점과 식료품 판매점 경계가 큼 | exclude |
| dining_hall | 식당 / 행사장 | 학교/단체 급식/행사 공간처럼 보일 수 있어 일반 식당과 다름 | exclude |
| dining_room | 식당 / 주거 | 가정집 식사 공간과 혼동 가능 | exclude |
| ice_cream_parlor | 카페 / 쇼핑 | 디저트 매장, 제품 close-up, 판매점 이미지가 섞임 | exclude |
| sushi_bar | 식당 / 술집 | 식당으로 쓸 수 있는 샘플도 있으나 바 좌석/술집 단서와 섞임 | exclude |
| discotheque | 술집 / 문화 | 클럽, 공연, 무대, 조명 이미지가 섞여 술집 단서가 안정적이지 않음 | exclude |
| wet_bar | 술집 / 주거 / 호텔 | 실제 샘플은 욕실, 주방, 가정용 바 설비가 많아 서비스 술집과 불일치 | exclude |
| auditorium | 문화 / 행사장 | 문화 시설로 볼 수 있지만 컨퍼런스/강당/행사장으로 넓게 섞임 | exclude |
| library_outdoor | 문화 / 건물 외관 | 건물 외관 중심이면 문화 활동 공간 단서가 약함 | exclude |
| museum_outdoor | 문화 / 건물 외관 | 박물관 내부 경험보다 외관/관광지 단서가 우세할 수 있음 | exclude |
| music_studio | 문화 / 작업공간 | 문화 향유 장소보다 작업실/녹음실 단서가 강함 | exclude |
| lawn | 공원 / 일반 녹지 | 잔디밭, 주택 마당, 골프장, 조경 등 generic green space로 확산됨 | exclude |
| roof_garden | 공원 / 건물 공간 | 공원보다 건물 부속 공간으로 보일 가능성이 큼 | exclude |
| shopfront | 쇼핑 / 거리 외관 | 상점 내부 쇼핑 경험보다 거리/건물 외관 단서가 강함 | exclude |
| ski_resort | 운동 / 여행 | 운동보다 숙박/관광/리조트 단서가 섞임 | exclude |
| stadium_baseball | 운동 / 관람장 | 직접 운동 장소보다 관중석/경기장 규모 단서가 강함 | exclude |
| stadium_football | 운동 / 관람장 | 직접 운동 장소보다 관중석/경기장 규모 단서가 강함 | exclude |
| stadium_soccer | 운동 / 관람장 | 직접 운동 장소보다 관중석/경기장 규모 단서가 강함 | exclude |
| stage_indoor | 문화 / 공연 세부 | 공연장 전체보다 무대 일부 장면에 치우침 | exclude |
| stage_outdoor | 문화 / 행사장 | 야외 행사/무대/축제 단서가 섞임 | exclude |
| swimming_hole | 운동 / 자연 | 수영장보다 자연 물놀이 장소로 보임 | exclude |
| water_park | 운동 / 공원 / 여행 | 운동보다 레저/테마파크 단서가 강함 | exclude |

## 재수집 시 주의할 Label

현재 0장인 label 중 아래 항목은 샘플이 확보되면 다시 육안 검토가 필요합니다.

| Places365 label | 후보 카테고리 | 주의점 |
| --- | --- | --- |
| bazaar_outdoor | 쇼핑 / 거리 / 관광 | 야외 시장은 쇼핑이지만 관광지/거리 장면으로도 보일 수 있음 |
| market_outdoor | 쇼핑 / 거리 / 관광 | 노점, 거리, 축제 이미지와 혼동 가능 |
| athletic_field_outdoor | 운동 / 공원 | 빈 들판이나 학교 운동장 풍경처럼 보일 수 있음 |
| ice_skating_rink_outdoor | 운동 / 공원 / 여행 | 야외 관광 시설 단서가 섞일 수 있음 |
| swimming_pool_outdoor | 운동 / 여행 / 숙박 | 호텔/리조트 수영장과 혼동 가능 |
| volleyball_court_outdoor | 운동 / 공원 / 해변 | 해변/공원 레저 장면과 혼동 가능 |

## 최종 권장안

- `bookstore`는 `문화`로 옮기기보다 v2에서는 제외합니다. 서비스가 “서점 방문”을 독립적으로 중요하게 다루기 전까지는 쇼핑/문화 양쪽 학습 신호를 흐릴 가능성이 큽니다.
- `cafeteria`는 카페로 넣지 않습니다. 식당으로도 일부 가능하지만 구내식당/행사 식사 공간 noise가 커서 제외 유지가 안전합니다.
- `beer_garden`, `food_court`, `restaurant_patio`, `amphitheater`는 label 자체보다 샘플 품질이 문제입니다. 전체 제거보다 sample-level filtering이 데이터 손실을 줄입니다.
- `기타`는 v2 Places365 데이터셋에 포함되어 있지 않습니다. unknown/negative class가 필요하면 별도 수집 기준을 세워야 합니다.
