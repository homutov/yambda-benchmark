## SANSA

Training SANSA on Yambda-500M with Listens, on Yambda-5B with Likes and on Yambda-5B with Listens is infeasible
due to the memory explosion during sparse matrix multiplication.

For example, given the base weights density of \\(d=5 \cdot 10^{-5}\\), the sparse matrix multiplication for Yambda-500M with Listens requires nearly
\\(N_{events} \cdot |I|^2 \cdot d = 0.5 \cdot 10^9 \cdot (3 \cdot 10^6) ^ 2 \cdot 5 \cdot 10^{-5} = 2.3 \cdot 10^{17}\\) operations.

One may observe that the good solution is to try to reduce the weights density. However given the results on Yambda-50M with Listens, further reducing density would collapse the model’s capacity, it becomes obvious that it is just practically futile. 

Unfortunately the main SANSA repository is not optimized for large datasets, so even on the Yambda-50M with Listens at least 100GB of RAM is required.
