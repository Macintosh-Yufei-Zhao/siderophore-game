from numba import cuda
print("可用GPU数量:", len(cuda.gpus))