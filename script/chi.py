import re 
import sys
# import matplotlib.pyplot as plt
# from matplotlib import rc
import numpy as np
import glob

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # main codes
    # hList = glob.glob('H_*')
    hList = np.arange(0.00, 2.04, 0.04)
 
    Egs = np.zeros((len(hList), 2), dtype=float)
    EgsD1 = np.zeros((len(hList)-1, 2), dtype=float)
    EgsD2 = np.zeros((len(hList)-2, 2), dtype=float)
    
    EgsD1[:, 0] = np.arange(0.02, 2.00, 0.04)
    EgsD2[:, 0] = np.arange(0.04, 2.00, 0.04)

    for i,h in enumerate(hList):
        filename = "H_" + "{:.2f}".format(h) + "/logfile.log";
        istr = open(filename, 'r')
        logger = istr.readlines()
        istr.close()
        
        Nx = re.match("LLX = (\d+)", logger[1])
        Ny = re.match("LLY = (\d+)", logger[2])
        N = int(Nx.groups()[0]) * int(Ny.groups()[0]) * 2
        Eg = -99999.9
        match = re.match("([+-]?\d+\.\d+)", logger[-8])
        print(logger[-8])
        if match:
            Eg = float(match.groups()[0])

        Egs[i, 0] = h
        Egs[i, 1] = Eg
        
        EgsD1[:, 1] = derivative(Egs[:, 1], Egs[:, 0])
        EgsD2[:, 1] = derivative(EgsD1[:, 1], EgsD1[:, 0])
        
        print(h)
    
        
# 
#     fig, ax = plt.subplots(1, 1,  figsize=(9,6))  # 1 row 1 col
#     
#     ax.plot(EgsD2[:, 0], -EgsD2[:, 1], marker='s', label=r"$K_z=1.0$")              
#     
#     ax.legend(loc='best', fontsize=18)
#     ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
#     # plt.yscale('log')
#     # plt.xscale('log')
# 
# 
#     ax.set_xlabel(r"$h$", fontsize=18)
#     ax.set_ylabel(r"$\chi(h)$", fontsize=18)
#     # plt.show()
#     plt.savefig("Egs.pdf")
    print(h)

    printfArray(np.abs(EgsD2), "Chi.dat")




def printfArray(A, filename, transpose = False):
    file = open(filename, "w")
    try:
        col = A.shape[1]
    except IndexError:
        A = A.reshape(-1, 1) 
    
    row = A.shape[0]
    col = A.shape[1]

    if transpose == False:
        for i in range(row):
            for j in range(col - 1):
                file.write(str(A[i, j]) + " ")
            file.write(str(A[i, col - 1]))  # to avoid whitespace at the end of line
            file.write("\n")
    elif transpose == True:
        for i in range(col):
            for j in range(row - 1):
                file.write(str(A[j, i]) + " ")
            file.write(str(A[row - 1, i]))
            file.write("\n")
    else:
        raise ValueError("3rd input must be Bool")
    file.close()


def derivative(Ys, Xs):
    derivatives = []
    for i in range(len(Xs) - 1):
        deltaY = Ys[i+1] - Ys[i]
        deltaX = Xs[i+1] - Xs[i]
        derivatives.append(deltaY / deltaX)
    return np.array(derivatives)
    

def toCplx(s):
    repart = float(s.split(",")[0].split("(")[1])
    impart = float(s.split(",")[1].split(")")[0])
    return complex(repart,impart)


def readfArray(str, Complex = False):
    file = open(str,'r')
    lines = file.readlines()
    file.close()

    # Determine shape:
    row = len(lines)
    testcol = lines[0].strip("\n").rstrip().split()
    col = len(testcol)  # rstip to rm whitespace at the end 

    m = np.zeros((row, col))
    for i in range(row):
        if lines[i] != "\n":
            line = lines[i].strip("\n").rstrip().split()
            # print(line)
            for j in range(col):
                val = float(line[j])
                m[i, j] = val
    return m



if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)