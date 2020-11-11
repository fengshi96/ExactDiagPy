import numpy as np
import sys


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


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        pass
