import numpy as np
import scipy.sparse as sp
from src.Parameter import Parameter
from src.Lattice import Lattice
from src.Dofs import Dofs
from src.Helper import matprint


class Hamiltonian:

	def __init__(self, para):

		self.Model = para.Model

		self.Hx = para.Hx  # uniform field in x
		self.Hy = para.Hy
		self.Hz = para.Hz

		self.Kxx = para.Kxx  # coupling strength of x-bond
		self.Kyy = para.Kyy
		self.Kzz = para.Kzz

		self.KxxPair_ = np.zeros(())  # pairwise non-zero coupling \\
		self.KyyPair_ = np.zeros(())  # 1st and 2nd cols are site indices
		self.KzzPair_ = np.zeros(())

		self.Kxxcoef_ = np.zeros(())  # pairwise non-zero coupling strength
		self.Kyycoef_ = np.zeros(())
		self.Kzzcoef_ = np.zeros(())

		if self.Model == "Kitaev":
			self.Nsite = para.LLX * para.LLY * 2
			self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
			self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
			self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
			self.Ham = self.BuildKitaev(para)

		elif self.Model == "Heisenberg":
			self.Nsite = para.LLX * para.LLY
			self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
			self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
			self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
			self.Ham = self.BuildHeisenberg(para)

		elif self.Model == "Hubbard":
			self.Nsite = para.LLX * para.LLY
			self.tGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
			self.Ham = self.BuildHubbard(para)

		else:
			pass

	def BuildKitaev(self, para):

		lat = Lattice(para)

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

		self.KxxPair_, self.Kxxcoef_, Nxbonds = PairConstructor(self.KxxGraph_, self.Nsite)
		self.KyyPair_, self.Kyycoef_, Nybonds = PairConstructor(self.KyyGraph_, self.Nsite)
		self.KzzPair_, self.Kzzcoef_, Nzbonds = PairConstructor(self.KzzGraph_, self.Nsite)


		# ---------------------Build Hamiltonian as Sparse Matrix-------------------

		print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")
		Spins = Dofs("SpinHalf")
		Sx = Spins.Sx
		Sy = Spins.Sy
		Sz = Spins.Sz
		I = Spins.I

		Ham = sp.eye(2 ** self.Nsite, dtype=complex) * 0
		Hamx = sp.eye(2 ** self.Nsite, dtype=complex) * 0
		Hamy = sp.eye(2 ** self.Nsite, dtype=complex) * 0
		Hamz = sp.eye(2 ** self.Nsite, dtype=complex) * 0

		for i in range(0, Nxbonds):
			ia = self.KxxPair_[i, 0]
			ib = self.KxxPair_[i, 1]
			coef = self.Kxxcoef_[i]

			ida = sp.eye(2 ** ia)
			idm = sp.eye(2 ** (ib - ia - 1))
			idb = sp.eye(2 ** (self.Nsite - ib - 1))
			tmp1 = sp.kron(ida, Sx)
			tmp2 = sp.kron(tmp1, idm)
			tmp3 = sp.kron(tmp2, Sx)
			tmp4 = sp.kron(tmp3, idb)
			Hamx += tmp4 * coef

		for i in range(0, Nybonds):
			ia = self.KyyPair_[i, 0]
			ib = self.KyyPair_[i, 1]
			coef = self.Kyycoef_[i]

			ida = sp.eye(2 ** ia)
			idm = sp.eye(2 ** (ib - ia - 1))
			idb = sp.eye(2 ** (self.Nsite - ib - 1))
			tmp1 = sp.kron(ida, Sy)
			tmp2 = sp.kron(tmp1, idm)
			tmp3 = sp.kron(tmp2, Sy)
			tmp4 = sp.kron(tmp3, idb)
			Hamy += tmp4 * coef

		for i in range(0, Nzbonds):
			ia = self.KzzPair_[i, 0]
			ib = self.KzzPair_[i, 1]
			coef = self.Kzzcoef_[i]

			ida = sp.eye(2 ** ia)
			idm = sp.eye(2 ** (ib - ia - 1))
			idb = sp.eye(2 ** (self.Nsite - ib - 1))
			tmp1 = sp.kron(ida, Sz)
			tmp2 = sp.kron(tmp1, idm)
			tmp3 = sp.kron(tmp2, Sz)
			tmp4 = sp.kron(tmp3, idb)
			Hamz += tmp4 * coef

		Ham = Hamx + Hamy + Hamz

		# --------------------------- Add external field -------------------------

		for i in range(0, self.Nsite):
			ida = sp.eye(2 ** i)
			idb = sp.eye(2 ** (self.Nsite - i - 1))
			Ham += sp.kron(ida, sp.kron(Sx, idb)) * self.Hx
			Ham += sp.kron(ida, sp.kron(Sy, idb)) * self.Hy
			Ham += sp.kron(ida, sp.kron(Sz, idb)) * self.Hz

		return Ham

	def BuildHeisenberg(self, para):
		lat = Lattice(para)

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

		self.KxxPair_, self.Kxxcoef_, Nxbonds = PairConstructor(self.KxxGraph_, self.Nsite)
		self.KyyPair_, self.Kyycoef_, Nybonds = PairConstructor(self.KyyGraph_, self.Nsite)
		self.KzzPair_, self.Kzzcoef_, Nzbonds = PairConstructor(self.KzzGraph_, self.Nsite)

	def BuildHubbard(self, para):
		pass


# --------------------------- Functions -------------------------
# --------------------------- Functions -------------------------
# --------------------------- Functions -------------------------
def PairConstructor(Graph_, Nsite):

	bonds = int(np.count_nonzero(Graph_) / 2)  # Number of non-zero bonds. Only half of matrix elements is needed
	PairInd_ = np.zeros((bonds, 2))  # Indices of pairs of sites that have non-zero coupling
	PairCoef_ = np.zeros(bonds)  # coupling constant of pairs

	# extract non-zero coupling pairs and their magnitude
	counter = 0
	for i in range(0, Nsite):
		for j in range(i, Nsite):
			if Graph_[i, j] != 0:
				PairInd_[counter, 0] = i
				PairInd_[counter, 1] = j
				PairCoef_[counter] = Graph_[i, j]
				counter += 1

	return PairInd_, PairCoef_, bonds


para = Parameter("../input.inp")
Ham = Hamiltonian(para)
