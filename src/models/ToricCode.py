import scipy.sparse as sp
from src.Hamiltonian import Hamiltonian, PairConstructor, TwoSpinOps
from src.Dofs import Dofs
from src.Helper import matprint


def SzBuild(site, Nsite, qm="SpinHalf"):
    """
    Build local spin z operator in full Hilbert space
    """
    if qm not in ["SpinHalf", "SpinOne"]:
        raise ValueError("Must be SpinHalf or SpinOne")

    dof = Dofs(qm)
    sz = dof.Sz
    hilbsize = dof.hilbsize

    ida = sp.eye(hilbsize ** site)
    idb = sp.eye(hilbsize ** (Nsite - site - 1))
    Sz = sp.kron(ida, sp.kron(sz, idb))
    return Sz


def SxBuild(site, Nsite, qm="SpinHalf"):
    """
    Build local spin z operator in full Hilbert space
    """
    if qm not in ["SpinHalf", "SpinOne"]:
        raise ValueError("Must be SpinHalf or SpinOne")

    dof = Dofs(qm)
    sx = dof.Sx
    hilbsize = dof.hilbsize

    ida = sp.eye(hilbsize ** site)
    idb = sp.eye(hilbsize ** (Nsite - site - 1))
    Sx = sp.kron(ida, sp.kron(sx, idb))
    return Sx


class ToricCode(Hamiltonian):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)
        self.Lat = Lat
        self.Nsite = Lat.LLX * Lat.LLY
        self.Ham = self.BuildToricCode()

    def BuildToricCode(self):
        Lat = self.Lat
        Ham = sp.eye(2 ** self.Lat.Nsite) * 0
        for i in range(self.Lat.Nsite):
            if i % 2 == 0:  # even for star
                ul = i
                ur = Lat.nn_[ul, 2]
                br = Lat.nn_[ur, 0]
                bl = Lat.nn_[ul, 0]
                Ham -= SxBuild(ul, self.Nsite) * SxBuild(ur, self.Nsite) \
                       * SxBuild(br, self.Nsite) * SxBuild(bl, self.Nsite) * (2 ** 4)
                print("even for star:", ul, ur, br, bl)
            else:  # odd for plaquette
                ul = i
                ur = Lat.nn_[ul, 2]
                br = Lat.nn_[ur, 0]
                bl = Lat.nn_[ul, 0]
                Ham -= SzBuild(ul, self.Nsite) * SzBuild(ur, self.Nsite) \
                       * SzBuild(br, self.Nsite) * SzBuild(bl, self.Nsite) * (2 ** 4)
                print("odd for plaquette:", ul, ur, br, bl)
        return Ham


if __name__ == '__main__':
    from src.Parameter import Parameter
    from src.Lattice import Lattice
    from src.Observ import matele
    import primme

    Para = Parameter("../../input.inp")  # import parameters from input.inp
    Lat = Lattice(Para)  # Build lattice
    Hamil = ToricCode(Lat, Para).Ham
    evals, evecs = primme.eigsh(Hamil, Para.parameters["Nstates"], tol=Para.parameters["tolerance"], which='SA')
    print(evals)

    # Observe: if <|As|> = 1?
    ul = 6
    ur = 7
    br = 12
    bl = 11
    Nsite = 20
    As = SxBuild(ul, Nsite) * SxBuild(ur, Nsite) * SxBuild(br, Nsite) * SxBuild(bl, Nsite) * (2 ** 4)
    E_As = matele(evecs[:, 0], As, evecs[:, 0])
    print(E_As)

    # Observe: if <|Bp|> = 1?
    ul = 7
    ur = 8
    br = 13
    bl = 12
    Nsite = 20
    Bp = SzBuild(ul, Nsite) * SzBuild(ur, Nsite) * SzBuild(br, Nsite) * SzBuild(bl, Nsite) * (2 ** 4)
    E_Bp = matele(evecs[:, 0], Bp, evecs[:, 0])
    print(E_Bp)

