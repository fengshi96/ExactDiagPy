import numpy as np


def matprint(A):
    row = A.shape[0]
    col = A.shape[1]
    for i in range(row):
        print("\n")
        for j in range(col):
            print(A[i, j], end='\t')
    print("\n")


def vecprint(A):
    row = len(A)
    for i in range(row):
        print(A[i], end='\t')
    print("\n")


def matprintos(A, filename):
    file = open(filename, "w")
    row = A.shape[0]
    col = A.shape[1]
    for i in range(row):
        for j in range(col):
            file.write(str(A[i, j]) + "\t")
        file.write("\n")
    file.close()


def sort(evals, evecs):
    index_ascend = np.argsort(evals)
    evals_sorted = evals[index_ascend]
    evecs_sorted = evecs[:, index_ascend]
    return evals_sorted, evecs_sorted
