# Gemma Direct Category vs Split Classifier Experiment

- created_at: 2026-05-10T22:14:09
- input_jsonl: `data/category_classifier/places365_v1/processed_service_prompt_smoke/test.jsonl`
- rows: 7
- direct_labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원, 기타
- direct_prompt: `experiments/gemma_category_compare/prompt_json_strict.txt`
- direct_prompt_hash: `bf5d0a2cc063`
- output_dir: `/home/cvlab/Desktop/Yoon/LighTrip-AI/experiments/gemma_category_compare/results/boundary_v2_smoke`

## Metric Summary

| method | rows | accuracy | macro_f1 | mean_sec | p95_sec | parse_success | consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split_pipeline | 7 | 1.0000 | 1.0000 | 18.4781 | 19.9069 | - | - |

## Output Files

- `metrics.json`: 전체 수치와 classification report dict
- `*_predictions.jsonl`: method별 raw 예측 결과
- `reports/*_classification_report.txt`: 카테고리별 precision/recall/F1
- `reports/*_confusion_matrix.csv`: confusion matrix

## Recommended Read

- `classifier_existing_draft`는 저장된 Gemma 초안을 SVM에 넣은 빠른 기준선입니다.
- `gemma_direct`는 같은 이미지에서 Gemma가 초안과 카테고리를 동시에 출력한 결과입니다.
- `split_pipeline`은 `--run-split-pipeline`을 켰을 때만 생성되며 실제 운영형 end-to-end와 가장 가깝습니다.
