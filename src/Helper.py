def matprint(A):
    row = A.shape[0]
    col = A.shape[1]
    for i in range(row):
        print("\n")
        for j in range(col):
            print(A[i, j], end='\t')
    print("\n")