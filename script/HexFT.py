import sys, math
sys.path.append('/Barn/Lab/ED_Python')
import numpy as np
from src.Parameter import Parameter
from src.Lattice import Lattice
from src.Helper import matprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.interpolate


def main(total, cmdargs):
    if total != 2:
        print(" ".join(str(x) for x in cmdargs))
        raise ValueError('Missing arguments')
    inputname = cmdargs[1]

    rawData = readfArray(inputname)
    data = rawData[0, 1:]

    Para = Parameter("input.inp")
    Lat = Lattice(Para)

    cMap = Lat.cMap
    cMapNew = cMap.copy()

    centerOffset = 11  # 0 -> centerOffset
    rc1, rc2 = cMap[centerOffset, :]

    cMapNew[:, 0] -= rc1
    cMapNew[:, 1] -= rc2
    print("\noffcentered rMap:")
    matprint(cMapNew)
    
    # to Euclidean basis
    # a = (1/2, sqrt{3}/2); b = (1, 0)
    # ria*a + rib*b = rix * x + riy * y
    for i in range(Lat.Nsite):
        rix = 0.5 * cMapNew[i, 0] + cMapNew[i, 1]
        riy = 0.5 * math.sqrt(3) * cMapNew[i, 0]
        cMapNew[i, 0] = rix
        cMapNew[i, 1] = riy

    print("\nEuclidean rMap:")
    matprint(cMapNew)

    print(data.shape, cMapNew.shape)

    # Begin FT
    n = 64  # 32 * 2
    outputdata = []
    S = np.zeros((3 * n, 3 * n))
    K1 = np.linspace(-3, 3, 3 * n)
    K2 = np.linspace(-3, 3, 3 * n)

    for i, k1 in enumerate(K1):
        for j, k2 in enumerate(K2):
            for si in range(Lat.Nsite):
                r1 = cMapNew[si, 0]
                r2 = cMapNew[si, 1]
                S[i, j] += math.e ** (2 * math.pi * complex(0, 1) * (r1 * k1 + r2 * k2)) * data[si]
            outputdata.append([k1, k2, S[i, j]])
    # matprint(S)
    output = np.vstack(outputdata)

    N = 4 * n
    xi = np.linspace(output[:, 0].min(), output[:, 0].max(), N)
    yi = np.linspace(output[:, 1].min(), output[:, 1].max(), N)
    zi = scipy.interpolate.griddata((output[:, 0], output[:, 1]), output[:, 2], (xi[None, :], yi[:, None]), method='cubic')


    # Plot
    fig = plt.figure(figsize=(7, 6))
    gs = gridspec.GridSpec(1, 1)

    ax0 = plt.subplot(gs[0])
    plt.xlabel('$k1$', size=18)
    plt.ylabel('$k2$', size=18, labelpad=0)
    plt.subplots_adjust(left=0.18, bottom=0.23, right=0.95, top=0.96, wspace=None, hspace=None)
    levels = np.linspace(0,0.5,100)

    plt.contourf(xi, yi, zi, levels=levels, cmap=plt.cm.get_cmap('seismic'), extend="both")
    plt.contourf(xi, yi, zi, levels=levels, cmap=plt.cm.get_cmap('seismic'), extend="both")
    cbar = plt.colorbar()
    plt.savefig("testFigure.pdf", dpi=600, bbox_inches='tight')
    plt.close()








def readfArray(str, Complex = False):
    def toCplx(s):
        if "j" in s:
            return complex(s)
        else:
            repart = float(s.split(",")[0].split("(")[1])
            impart = float(s.split(",")[1].strip("j").split(")")[0])
            return complex(repart,impart)
    
    file = open(str,'r')
    lines = file.readlines()
    file.close()

    # Determine shape:
    row = len(lines)
    testcol = lines[0].strip("\n").rstrip().split()
    col = len(testcol)  # rstip to rm whitespace at the end 

    if Complex:
        m = np.zeros((row, col), dtype=complex)
        for i in range(row):
            if lines[i] != "\n":
                line = lines[i].strip("\n").rstrip().split()
                # print(line)
                for j in range(col):
                    val = toCplx(line[j])
                    m[i, j] = val
        
    else:
        m = np.zeros((row, col), dtype=float)
        for i in range(row):
            if lines[i] != "\n":
                line = lines[i].strip("\n").rstrip().split()
                print(line)
                for j in range(col):
                    val = float(line[j])
                    m[i, j] = val
    return m



if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
