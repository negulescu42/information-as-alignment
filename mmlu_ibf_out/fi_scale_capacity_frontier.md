# Cell 20b — High-Capacity Fixed-σ Scale Frontier

## Summary

| N | target | base | control | epoch | centers | cap | growth/rule | vmax | read ms/item | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1000 | 0.995 | 0.005 | 1.000 | 1 | 8369 | 9882 | 1.987 | 7.520 | 0.586 | no |
| 3000 | 0.980 | 0.019 | 1.000 | 1 | 12270 | 14882 | 1.963 | 7.520 | 0.757 | no |
| 5000 | 0.971 | 0.027 | 1.000 | 1 | 16102 | 19882 | 1.944 | 7.520 | 0.834 | no |
| 10000 | 0.951 | 0.045 | 1.000 | 1 | 25459 | 32382 | 1.908 | 7.520 | 1.071 | no |
| 20000 | 0.931 | 0.063 | 1.000 | 1 | 43645 | 57382 | 1.863 | 7.520 | 1.337 | no |
| 50000 | 0.886 | 0.101 | 1.000 | 1 | 94634 | 132382 | 1.765 | 7.520 | 2.096 | no |

## Interpretation

This run tests whether the earlier 10k degradation was caused by the hard 20k center cap. Capacity is lifted dynamically while σ remains fixed at the canonical operating geometry.
