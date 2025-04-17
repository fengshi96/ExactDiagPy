import sys, math
import numpy as np
import time
import primme
from src.Parameter import Parameter
from src.Observ import Observ
from src.Lattice import Lattice
from src.Helper import Logger, sort, hd5Storage, matprint, printfArray
import scipy.sparse.linalg
from src.models.Kitaev_Ladder import Kitaev_Ladder
import h5py
from src.Dofs import Dofs
from src.Observ import overlap, Observ, matele
from src.observe.ladder_dynamics import KitaevLadderObserv

pi = np.pi


def main(total, cmdargs):
    param = Parameter("input.inp")
    lat = Lattice(param)
    test = Kitaev_Ladder(lat, param)

    # test measuring fluxes 
    outputname = "dataSpec.hdf5"
    inputname = "input.inp"

    para = Parameter(inputname)
    if para.parameters['Method'] == "ED":
        para.parameters["Nstates"] = 2 ** lat.Nsite
    Lat = Lattice(para)
    ob = KitaevLadderObserv(Lat, para)
    print(para.parameters["Model"])
    dof = Dofs("SpinHalf")  # default spin-1/2

    # ------- Read dataSpec file -------
    rfile = h5py.File(outputname, 'r')
    evalset = rfile["3.Eigen"]["Eigen Values"]
    evecset = rfile["3.Eigen"]["Wavefunctions"]

    # extract eigen value and eigen vector from HDF5 rfile
    evals = np.zeros(para.parameters["Nstates"], dtype=float)
    evecs = np.zeros((dof.hilbsize ** Lat.Nsite, para.parameters["Nstates"]), dtype=complex)
    evalset.read_direct(evals)
    evecset.read_direct(evecs)
    rfile.close()
    gs = evecs[:, 0]


    # apply perturbation locally
    op1 = ob.LSzBuild(Lat.LLX, pauli=True)
    op2 = ob.LSzBuild(Lat.LLX + 1, pauli=True)
    print("Perturbation at", Lat.LLX) 
    psi_init = op1 @ gs



    energy_densities_gs = np.zeros(Lat.Nsite // 4, dtype=float)
    for i in np.arange(0, Lat.Nsite//4, 1):
        energy_densities_gs[i] = matele(gs, ob.buildenergydensity(i * 4 + 2), gs).real
    average_gsenergy_per_site = np.sum(energy_densities_gs) / (Lat.Nsite // 4)

    energy_densities_pert = np.zeros(Lat.Nsite // 4, dtype=float)
    for i in np.arange(0, Lat.Nsite//4, 1):
        energy_densities_pert[i] = matele(psi_init, ob.buildenergydensity(i * 4 + 2), psi_init).real
    average_pertenergy_per_site = np.sum(energy_densities_pert) / (Lat.Nsite // 4) - average_gsenergy_per_site


    psi_init = psi_init / np.linalg.norm(psi_init)
    c_n = evecs.conjugate().T @ psi_init        # shape (Nstates,)


    t_max   = 800             # maximum real time you want (units of ħ=1)
    Nt      = 4000                # number of time points (Nt–1 steps of dt)
    times   = np.linspace(0.0, t_max, Nt)

    # arrays to store entropy vs time
    # SvN = np.zeros_like(times, dtype=float)
    Engs = np.zeros((Nt, Lat.Nsite // 4), dtype=float)
    energy_op_list = [ob.buildenergydensity(i * 4 + 2) for i in range(Lat.Nsite // 4)]



    for it, t in enumerate(times):

        # phase factors  e^{-i E_n t}
        phase_t = np.exp(-1j * evals * t)    # shape (Nstates,)

        # reconstruct |psi(t)> = sum_n c_n e^{-iE_n t} |n>
        psi_t = evecs @ (c_n * phase_t)    # shape (65_536,)
        psi_t = psi_t / np.linalg.norm(psi_t)

        for i, op in enumerate(energy_op_list):
            Engs[it, i] = matele(psi_t, op, psi_t).real - average_gsenergy_per_site

        if it % 50 == 0:  # light progress report
            print(f" t = {t:6.2f}")



    printfArray(Engs, "Engs.txt")

    import matplotlib.pyplot as plt
    figure = plt.figure()
    plt.plot(times, Engs[:, 1], '-o', ms=3, label='m')
    # plt.plot(times, Engs[:, 0], '-o', ms=3, label='l')
    # plt.plot(times, Engs[:, 2], '-o', ms=2, label='r')

    # late_time_mean_mid = np.mean(Engs[400:, 1])
    max_time_to_plot = 10
    plt.hlines(average_pertenergy_per_site, xmin=0, xmax=max_time_to_plot, color='black', lw=0.5, ls='--')
    plt.legend(loc = 'upper right')
    plt.xlabel(r'$t$')
    plt.xlim(xmin=0, xmax=max_time_to_plot)
    plt.ylim(ymin=average_pertenergy_per_site * 0.8, ymax=3 * average_pertenergy_per_site)
    plt.yscale('log')
    # plt.ylabel(r'$S_{\mathrm{vN}}(t)$ (first 8 spins)')
    plt.ylabel(r'$\langle E \rangle$ ')
    plt.tight_layout()
    # plt.show()
    figure.savefig("time_evo.pdf", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
