import numpy as np

# this function receives a txt file and gives a np matrix and its size
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    matrix_list = [line.strip().split() for line in lines]
    matrix_list = [[float(element) for element in row] for row in matrix_list]
    matrix_np = np.array(matrix_list)
    num_rows, num_cols = matrix_np.shape
    return matrix_np, num_rows, num_cols

def write_matrix(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%f', delimiter=' ')


        