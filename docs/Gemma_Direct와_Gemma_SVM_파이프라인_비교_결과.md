# Gemma Direct Category vs Split Classifier Experiment

- created_at: 2026-05-11T15:35:27
- input_jsonl: `data_places365_2/processed/test.jsonl`
- rows: 339
- direct_labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원
- gemma_prompt: `configs/draft_prompt_boundary_v2.txt`
- direct_prompt: `configs/draft_prompt_boundary_v2.txt`
- direct_prompt_hash: `5907265f6bd9`
- output_dir: `/home/cvlab/Desktop/Yoon/LighTrip-AI/experiments/gemma_category_compare/results/recompare_places365_2_boundary_v2_balanced_no_etc`

## Metric Summary

| method | rows | accuracy | macro_f1 | mean_sec | p95_sec | parse_success | consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| classifier_existing_draft | 339 | 0.8555 | 0.8380 | 0.0005 | 0.0006 | - | - |
| gemma_direct | 339 | 0.8614 | 0.8471 | 2.2423 | 2.5601 | 1.0000 | 1.0000 |
| split_pipeline | 339 | 0.7788 | 0.7573 | 1.8612 | 2.1931 | - | - |

## Output Files

- `metrics.json`: 전체 수치와 classification report dict
- `*_predictions.jsonl`: method별 raw 예측 결과
- `reports/*_classification_report.txt`: 카테고리별 precision/recall/F1
- `reports/*_confusion_matrix.csv`: confusion matrix

## Recommended Read

- `classifier_existing_draft`는 저장된 Gemma 초안을 SVM에 넣은 빠른 기준선입니다.
- `gemma_direct`는 같은 이미지에서 Gemma가 초안과 카테고리를 동시에 출력한 결과입니다.
- `split_pipeline`은 `--run-split-pipeline`을 켰을 때만 생성되며 실제 운영형 end-to-end와 가장 가깝습니다.
