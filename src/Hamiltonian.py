import numpy as np
import scipy.sparse as sp
from src.Dofs import Dofs
from src.Helper import matprint


class Hamiltonian:

	def __init__(self, Lat):

		self.Lat = Lat
		self.Model = Lat.Model

		self.Hx = Lat.Hx  # uniform field in x
		self.Hy = Lat.Hy
		self.Hz = Lat.Hz

		self.Kxx = Lat.Kxx  # coupling strength of x-bond
		self.Kyy = Lat.Kyy
		self.Kzz = Lat.Kzz

		self.KxxPair_ = np.zeros(())  # pairwise non-zero coupling \\
		self.KyyPair_ = np.zeros(())  # 1st and 2nd cols are site indices
		self.KzzPair_ = np.zeros(())

		self.Kxxcoef_ = np.zeros(())  # pairwise non-zero coupling strength
		self.Kyycoef_ = np.zeros(())
		self.Kzzcoef_ = np.zeros(())

		if self.Model == "Kitaev":
			self.Nsite = Lat.LLX * Lat.LLY * 2
			self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Ham = self.BuildKitaev()

		elif self.Model == "Heisenberg":
			self.Nsite = Lat.LLX * Lat.LLY
			self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Ham = self.BuildHeisenberg()

		elif self.Model == "Hubbard":
			self.Nsite = Lat.LLX * Lat.LLY
			self.tGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
			self.Ham = self.BuildHubbard()

		else:
			raise ValueError("Model not supported")

	def BuildKitaev(self):

		lat = self.Lat
		for i in range(0, self.Nsite):
			# Kxx_Conn
			j = lat.nn_[i, 0]
			if i < j and j >= 0:
				self.KxxGraph_[i, j] = self.Kxx
				self.KxxGraph_[j, i] = self.Kxx

			# Kyy_Conn
			j = lat.nn_[i, 1]
			if i < j and j >= 0:
				self.KyyGraph_[i, j] = self.Kyy
				self.KyyGraph_[j, i] = self.Kyy

			# Kzz_Conn
			j = lat.nn_[i, 2]
			if i < j and j >= 0:
				self.KzzGraph_[i, j] = self.Kzz
				self.KzzGraph_[j, i] = self.Kzz

		print("\nKxxGraph_:")
		matprint(self.KxxGraph_)
		print("\nKyyGraph_:")
		matprint(self.KyyGraph_)
		print("\nKzzGraph_:")
		matprint(self.KzzGraph_)

		self.KxxPair_, self.Kxxcoef_ = PairConstructor(self.KxxGraph_, self.Nsite)
		self.KyyPair_, self.Kyycoef_ = PairConstructor(self.KyyGraph_, self.Nsite)
		self.KzzPair_, self.Kzzcoef_ = PairConstructor(self.KzzGraph_, self.Nsite)

		# ---------------------Build Hamiltonian as Sparse Matrix-------------------

		print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")
		Spins = Dofs("SpinHalf")
		Sx = Spins.Sx
		Sy = Spins.Sy
		Sz = Spins.Sz
		I = Spins.I

		Hamx = TwoSpinOps(self.KxxPair_, self.Kxxcoef_, Sx, self.Nsite)
		Hamy = TwoSpinOps(self.KyyPair_, self.Kyycoef_, Sy, self.Nsite)
		Hamz = TwoSpinOps(self.KzzPair_, self.Kzzcoef_, Sz, self.Nsite)

		Ham = Hamx + Hamy + Hamz

		# --------------------------- Add external field -------------------------

		for i in range(0, self.Nsite):
			ida = sp.eye(2 ** i)
			idb = sp.eye(2 ** (self.Nsite - i - 1))
			Ham += sp.kron(ida, sp.kron(Sx, idb)) * self.Hx
			Ham += sp.kron(ida, sp.kron(Sy, idb)) * self.Hy
			Ham += sp.kron(ida, sp.kron(Sz, idb)) * self.Hz

		return Ham

	def BuildHeisenberg(self):
		lat = self.Lat

		for bond in range(0, lat.Number1neigh):
			for i in range(0, self.Nsite):
				j = lat.nn_[i, bond]
				if i < j:
					# Kxx_ij * S_i^x S_j^x
					self.KxxGraph_[i, j] = self.Kxx
					self.KxxGraph_[j, i] = self.Kxx

					# Kyy_ij * S_i^y S_j^y
					self.KyyGraph_[i, j] = self.Kyy
					self.KyyGraph_[j, i] = self.Kyy

					# Kzz_ij * S_i^z S_j^z
					self.KzzGraph_[i, j] = self.Kzz
					self.KzzGraph_[j, i] = self.Kzz

		print("\nKxxGraph_:")
		matprint(self.KxxGraph_)
		print("\nKyyGraph_:")
		matprint(self.KyyGraph_)
		print("\nKzzGraph_:")
		matprint(self.KzzGraph_)

		self.KxxPair_, self.Kxxcoef_ = PairConstructor(self.KxxGraph_, self.Nsite)
		self.KyyPair_, self.Kyycoef_ = PairConstructor(self.KyyGraph_, self.Nsite)
		self.KzzPair_, self.Kzzcoef_ = PairConstructor(self.KzzGraph_, self.Nsite)

		# ---------------------Build Hamiltonian as Sparse Matrix-------------------

		print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")
		Spins = Dofs("SpinHalf")
		Sx = Spins.Sx
		Sy = Spins.Sy
		Sz = Spins.Sz
		I = Spins.I

		Hamx = TwoSpinOps(self.KxxPair_, self.Kxxcoef_, Sx, self.Nsite)
		Hamy = TwoSpinOps(self.KyyPair_, self.Kyycoef_, Sy, self.Nsite)
		Hamz = TwoSpinOps(self.KzzPair_, self.Kzzcoef_, Sz, self.Nsite)

		Ham = Hamx + Hamy + Hamz

		# --------------------------- Add external field -------------------------

		for i in range(0, self.Nsite):
			ida = sp.eye(2 ** i)
			idb = sp.eye(2 ** (self.Nsite - i - 1))
			Ham += sp.kron(ida, sp.kron(Sx, idb)) * self.Hx
			Ham += sp.kron(ida, sp.kron(Sy, idb)) * self.Hy
			Ham += sp.kron(ida, sp.kron(Sz, idb)) * self.Hz

		return Ham

	def BuildHubbard(self):
		pass


# --------------------------- Functions -------------------------
# --------------------------- Functions -------------------------
# --------------------------- Functions -------------------------
def PairConstructor(Graph_, Nsite):
	bonds = int(np.count_nonzero(Graph_) / 2)  # Number of non-zero bonds. Only half of matrix elements is needed
	PairInd_ = np.zeros((bonds, 2))  # Indices of pairs of sites that have non-zero coupling
	PairCoef_ = np.zeros(bonds)  # coupling constant of pairs

	# extract non-zero coupling pairs and their value
	counter = 0
	for i in range(0, Nsite):
		for j in range(i, Nsite):
			if Graph_[i, j] != 0:
				PairInd_[counter, 0] = i
				PairInd_[counter, 1] = j
				PairCoef_[counter] = Graph_[i, j]
				counter += 1

	return PairInd_, PairCoef_


def TwoSpinOps(PairInd_, PairCoef_, Dof, Nsite):
	Nbonds = len(PairCoef_)  # Number of non-zero bonds
	Hamtmp_ = sp.eye(2 ** Nsite, dtype=complex) * 0
	for i in range(0, Nbonds):
		ia = PairInd_[i, 0]
		ib = PairInd_[i, 1]
		coef = PairCoef_[i]

		ida = sp.eye(2 ** ia)
		idm = sp.eye(2 ** (ib - ia - 1))
		idb = sp.eye(2 ** (Nsite - ib - 1))

		tmp = sp.kron(sp.kron(sp.kron(sp.kron(ida, Dof), idm), Dof), idb)
		Hamtmp_ += tmp * coef

	return Hamtmp_


#para = Parameter("../input.inp")
#Ham = Hamiltonian(para)
