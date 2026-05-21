# Semantic Access Cost Profile

Local calibration for summary/PQ/full evidence operations using candidate-list sizes from the large IVF-PQ audit.

| view | median us | ratio to PQ | approx bytes at median n | benchmark cost |
| --- | ---: | ---: | ---: | ---: |
| summary_centroid | 0.500 | 0.021 | 768 | 0.00 |
| pq_codes | 24.300 | 1.000 | 28608 | 0.20 |
| full_vectors | 4.100 | 0.169 | 258048 | 0.58 |

Candidate counts: median 336, p90 731.
