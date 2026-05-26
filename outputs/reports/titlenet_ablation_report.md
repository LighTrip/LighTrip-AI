# TitLeNet Ablation Study

- trained: `True`
- results_csv: `/home/cvlab/Desktop/Yoon/LighTrip-AI/outputs/reports/titlenet_ablation_results.csv`

## Summary

| trial | group | model | act | init | params | size_mb | b1_ms | b64_ms | test_loss | test_ndcg@3 | test_ndcg@5 | max_color_share |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| titlenet | reference | titlenet | gelu | small_head | 183732 | 0.715 | 2.465 | 3.068 | 0.005902 | 0.989124 | 0.990784 | 0.381403 |
| titlenet_no_se | deletion | titlenet_no_se | gelu | small_head | 163920 | 0.639 | 1.457 | 2.727 | 0.007095 | 0.985883 | 0.987894 | 0.391075 |
| titlenet_no_residual | deletion | titlenet_no_residual | gelu | small_head | 80048 | 0.311 | 0.777 | 1.454 | 0.007800 | 0.983790 | 0.986127 | 0.396571 |
| titlenet_no_first_residual | deletion | titlenet_no_first_residual | gelu | small_head | 170856 | 0.664 | 1.703 | 2.036 | 0.007380 | 0.986805 | 0.988757 | 0.382502 |
| titlenet_no_middle_residual | deletion | titlenet_no_middle_residual | gelu | small_head | 161444 | 0.628 | 1.799 | 2.631 | 0.006229 | 0.988071 | 0.989834 | 0.382502 |
| titlenet_no_last_residual | deletion | titlenet_no_last_residual | gelu | small_head | 115212 | 0.448 | 1.418 | 2.725 | 0.006234 | 0.987440 | 0.989356 | 0.381183 |
| titlenet_no_last_extra_residual | deletion | titlenet_no_last_extra_residual | gelu | small_head | 149472 | 0.582 | 1.776 | 2.817 | 0.006005 | 0.988609 | 0.990466 | 0.382062 |
| titlenet_no_stem | stage_deletion | titlenet_no_stem | gelu | small_head | 182196 | 0.709 | 2.047 | 2.963 | 0.006195 | 0.987925 | 0.989892 | 0.382721 |
| titlenet_no_stage1 | stage_deletion | titlenet_no_stage1 | gelu | small_head | 170328 | 0.662 | 1.858 | 1.867 | 0.006137 | 0.988643 | 0.990400 | 0.380303 |
| titlenet_no_stage2 | stage_deletion | titlenet_no_stage2 | gelu | small_head | 160388 | 0.623 | 1.804 | 2.533 | 0.006328 | 0.987947 | 0.989836 | 0.384480 |
| titlenet_no_stage3 | stage_deletion | titlenet_no_stage3 | gelu | small_head | 113804 | 0.442 | 1.489 | 2.703 | 0.006547 | 0.986561 | 0.988846 | 0.386898 |

## Plots

![Latency](/home/cvlab/Desktop/Yoon/LighTrip-AI/outputs/reports/titlenet_ablation_latency.png)
![NDCG](/home/cvlab/Desktop/Yoon/LighTrip-AI/outputs/reports/titlenet_ablation_ndcg_curve.png)
![NDCG Delta](/home/cvlab/Desktop/Yoon/LighTrip-AI/outputs/reports/titlenet_ablation_ndcg5_delta.png)
![Paper Summary](/home/cvlab/Desktop/Yoon/LighTrip-AI/outputs/reports/titlenet_ablation_paper_summary.png)
![Residual Ablation](/home/cvlab/Desktop/Yoon/LighTrip-AI/outputs/reports/titlenet_residual_ablation_paper_summary.png)
![Stage Ablation](/home/cvlab/Desktop/Yoon/LighTrip-AI/outputs/reports/titlenet_stage_ablation_paper_summary.png)
