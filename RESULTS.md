# Differential test: numpy float32 vs verified IEEE-754 reference

Each row is a model configuration. The reference is the extracted
verified forward; numpy runs the same elementwise math with its own
reduction order. `abs-err` is over every logit of every sample.

| layers | d_model | heads | seq | vocab | samples | mean abs-err | max abs-err | next-token flips | nan |
|--------|---------|-------|-----|-------|---------|--------------|-------------|------------------|-----|
| 1 | 8 | 2 | 8 | 16 | 16 | 4.113e-09 | 4.434e-08 | 0/16 | 0 |
| 2 | 8 | 2 | 8 | 16 | 16 | 4.489e-09 | 3.021e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 8 | 16 | 16 | 5.920e-09 | 1.415e-07 | 0/16 | 0 |
| 4 | 8 | 2 | 8 | 64 | 16 | 5.570e-09 | 8.566e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 16 | 16 | 16 | 5.128e-09 | 4.502e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 32 | 16 | 16 | 4.989e-09 | 4.847e-08 | 0/16 | 0 |
| 4 | 16 | 4 | 8 | 16 | 16 | 1.233e-08 | 1.267e-07 | 0/16 | 0 |
| 4 | 32 | 8 | 8 | 16 | 16 | 3.657e-08 | 3.306e-07 | 0/16 | 0 |
| 4 | 64 | 8 | 8 | 16 | 16 | 1.123e-07 | 8.938e-07 | 0/16 | 0 |
| 8 | 8 | 2 | 8 | 16 | 16 | 6.623e-09 | 9.681e-08 | 0/16 | 0 |
