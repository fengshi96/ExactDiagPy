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

		if self.Model != "AKLT":
			self.KxxPair_ = np.zeros(())  # pairwise non-zero coupling \\
			self.KyyPair_ = np.zeros(())  # 1st and 2nd cols are site indices
			self.KzzPair_ = np.zeros(())

			self.Kxxcoef_ = np.zeros(())  # pairwise non-zero coupling strength
			self.Kyycoef_ = np.zeros(())
			self.Kzzcoef_ = np.zeros(())

			self.Kxx = Lat.Kxx  # coupling strength of x-bond
			self.Kyy = Lat.Kyy
			self.Kzz = Lat.Kzz

		# -----------------------------------------------------------------------

		if self.Model == "Kitaev":
			self.Nsite = Lat.LLX * Lat.LLY * 2
			self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Ham = self.BuildKitaev()

		elif self.Model == "Heisenberg_Honeycomb":
			self.Nsite = Lat.LLX * Lat.LLY * 2
			self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Ham = self.BuildHeisenberg()

		elif self.Model == "Heisenberg_Square":
			self.Nsite = Lat.LLX * Lat.LLY
			self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Ham = self.BuildHeisenberg()

		elif self.Model == "Hubbard":
			self.Nsite = Lat.LLX * Lat.LLY
			self.tGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Ham = self.BuildHubbard()

		elif self.Model == "AKLT":
			self.Nsite = Lat.LLX

			self.Kxx1Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Kyy1Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Kzz1Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)

			self.Kxx2Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Kyy2Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
			self.Kzz2Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)

			self.Kxx1Pair_ = np.zeros(())  # pairwise non-zero coupling \\
			self.Kyy1Pair_ = np.zeros(())  # 1st and 2nd cols are site indices
			self.Kzz1Pair_ = np.zeros(())

			self.Kxx2Pair_ = np.zeros(())
			self.Kyy2Pair_ = np.zeros(())
			self.Kzz2Pair_ = np.zeros(())

			self.Kxx1coef_ = np.zeros(())  # pairwise non-zero coupling strength
			self.Kyy1coef_ = np.zeros(())
			self.Kzz1coef_ = np.zeros(())

			self.Kxx2coef_ = np.zeros(())
			self.Kyy2coef_ = np.zeros(())
			self.Kzz2coef_ = np.zeros(())

			self.Ham = self.BuildAKLT()

		else:
			raise ValueError("Model not supported")

	def BuildAKLT(self):
		lat = self.Lat

		for bond in range(0, lat.Number1neigh):
			for i in range(0, self.Nsite):
				j = lat.nn_[i, bond]
				# print(j)
				if i < j and j >= 0:
					# Kxx1,2_ij
					self.Kxx1Graph_[i, j] = lat.Kxx1
					self.Kxx1Graph_[j, i] = lat.Kxx1
					self.Kxx2Graph_[i, j] = lat.Kxx2
					self.Kxx2Graph_[j, i] = lat.Kxx2

					# Kyy1,2_ij
					self.Kyy1Graph_[i, j] = lat.Kyy1
					self.Kyy1Graph_[j, i] = lat.Kyy1
					self.Kyy2Graph_[i, j] = lat.Kyy2
					self.Kyy2Graph_[j, i] = lat.Kyy2

					# Kzz1,2_ij
					self.Kzz1Graph_[i, j] = lat.Kzz1
					self.Kzz1Graph_[j, i] = lat.Kzz1
					self.Kzz2Graph_[i, j] = lat.Kzz2
					self.Kzz2Graph_[j, i] = lat.Kzz2

		print("\nKxx1Graph_:"); matprint(self.Kxx1Graph_)
		print("\nKyy1Graph_:"); matprint(self.Kyy1Graph_)
		print("\nKzz1Graph_:"); matprint(self.Kzz1Graph_)
		print("\nKxx2Graph_:"); matprint(self.Kxx2Graph_)
		print("\nKyy2Graph_:"); matprint(self.Kyy2Graph_)
		print("\nKzz2Graph_:"); matprint(self.Kzz2Graph_)

		self.Kxx1Pair_, self.Kxx1coef_ = PairConstructor(self.Kxx1Graph_, self.Nsite)
		self.Kyy1Pair_, self.Kyy1coef_ = PairConstructor(self.Kyy1Graph_, self.Nsite)
		self.Kzz1Pair_, self.Kzz1coef_ = PairConstructor(self.Kzz1Graph_, self.Nsite)

		self.Kxx2Pair_, self.Kxx2coef_ = PairConstructor(self.Kxx2Graph_, self.Nsite)
		self.Kyy2Pair_, self.Kyy2coef_ = PairConstructor(self.Kyy2Graph_, self.Nsite)
		self.Kzz2Pair_, self.Kzz2coef_ = PairConstructor(self.Kzz2Graph_, self.Nsite)

		# ---------------------Build Hamiltonian as Sparse Matrix-------------------

		print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")
		Spins = Dofs("SpinOne")
		if lat.Model == "TFIM":
			Spins = Dofs("SpinHalf")
		sx = Spins.Sx
		sy = Spins.Sy
		sz = Spins.Sz
		print("-------------------------------------------", sx.shape)

		Hamx1 = TwoSpinOps(self.Kxx1Pair_, self.Kxx1coef_, sx, sx, self.Nsite)
		Hamy1 = TwoSpinOps(self.Kyy1Pair_, self.Kyy1coef_, sy, sy, self.Nsite)
		Hamz1 = TwoSpinOps(self.Kzz1Pair_, self.Kzz1coef_, sz, sz, self.Nsite)

		Hamxxxx2 = TwoSpinOps(self.Kxx2Pair_, self.Kxx2coef_, sx * sx, sx * sx, self.Nsite)
		Hamxxyy2 = TwoSpinOps(self.Kyy2Pair_, self.Kyy2coef_, sy * sy, sy * sy, self.Nsite)
		Hamxxzz2 = TwoSpinOps(self.Kzz2Pair_, self.Kzz2coef_, sz * sz, sz * sz, self.Nsite)

		Hamxy2 = TwoSpinOps(self.Kxx2Pair_, self.Kxx2coef_, sx * sy, sx * sy, self.Nsite)
		Hamxz2 = TwoSpinOps(self.Kyy2Pair_, self.Kyy2coef_, sx * sz, sx * sz, self.Nsite)

		Hamyx2 = TwoSpinOps(self.Kzz2Pair_, self.Kzz2coef_, sy * sx, sy * sx, self.Nsite)
		Hamyz2 = TwoSpinOps(self.Kxx2Pair_, self.Kxx2coef_, sy * sz, sy * sz, self.Nsite)

		Hamzx2 = TwoSpinOps(self.Kyy2Pair_, self.Kyy2coef_, sz * sx, sz * sx, self.Nsite)
		Hamzy2 = TwoSpinOps(self.Kzz2Pair_, self.Kzz2coef_, sz * sy, sz * sy, self.Nsite)

		Ham = Hamx1 + Hamy1 + Hamz1 + Hamxy2 + Hamxz2 + Hamyx2 + Hamyz2 + Hamzx2 + Hamzy2
		Ham += Hamxxxx2 + Hamxxyy2 + Hamxxzz2

		# --------------------------- Add external field -------------------------

		hilsize = sx.shape[0]
		for i in range(0, self.Nsite):
			ida = sp.eye(hilsize ** i)
			idb = sp.eye(hilsize ** (self.Nsite - i - 1))
			Ham += sp.kron(ida, sp.kron(sx, idb)) * self.Hx
			Ham += sp.kron(ida, sp.kron(sy, idb)) * self.Hy
			Ham += sp.kron(ida, sp.kron(sz, idb)) * self.Hz

		#if lat.Model == "AKLT":
		Ham += sp.eye(Ham.shape[0]) * 2.0 / 3.0 * len(self.Kxx1coef_)
		return Ham

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
		sx = Spins.Sx
		sy = Spins.Sy
		sz = Spins.Sz

		Hamx = TwoSpinOps(self.KxxPair_, self.Kxxcoef_, sx, sx, self.Nsite)
		Hamy = TwoSpinOps(self.KyyPair_, self.Kyycoef_, sy, sy, self.Nsite)
		Hamz = TwoSpinOps(self.KzzPair_, self.Kzzcoef_, sz, sz, self.Nsite)

		Ham = Hamx + Hamy + Hamz

		# --------------------------- Add external field -------------------------

		for i in range(0, self.Nsite):
			ida = sp.eye(2 ** i)
			idb = sp.eye(2 ** (self.Nsite - i - 1))
			Ham += sp.kron(ida, sp.kron(sx, idb)) * self.Hx
			Ham += sp.kron(ida, sp.kron(sy, idb)) * self.Hy
			Ham += sp.kron(ida, sp.kron(sz, idb)) * self.Hz

		return Ham

	def BuildHeisenberg(self):
		lat = self.Lat

		for bond in range(0, lat.Number1neigh):
			for i in range(0, self.Nsite):
				j = lat.nn_[i, bond]
				# print(j)
				if i < j and j >= 0:
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

		#matprint(self.KxxGraph_-self.KzzGraph_); matprint(self.KxxGraph_-self.KyyGraph_)

		self.KxxPair_, self.Kxxcoef_ = PairConstructor(self.KxxGraph_, self.Nsite)
		self.KyyPair_, self.Kyycoef_ = PairConstructor(self.KyyGraph_, self.Nsite)
		self.KzzPair_, self.Kzzcoef_ = PairConstructor(self.KzzGraph_, self.Nsite)

		# ---------------------Build Hamiltonian as Sparse Matrix-------------------

		print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")
		Spins = Dofs("SpinHalf")
		sx = Spins.Sx
		sy = Spins.Sy
		sz = Spins.Sz

		Hamx = TwoSpinOps(self.KxxPair_, self.Kxxcoef_, sx, sx, self.Nsite)
		Hamy = TwoSpinOps(self.KyyPair_, self.Kyycoef_, sy, sy, self.Nsite)
		Hamz = TwoSpinOps(self.KzzPair_, self.Kzzcoef_, sz, sz, self.Nsite)

		Ham = Hamx + Hamy + Hamz

		# --------------------------- Add external field -------------------------

		for i in range(0, self.Nsite):
			ida = sp.eye(2 ** i)
			idb = sp.eye(2 ** (self.Nsite - i - 1))
			Ham += sp.kron(ida, sp.kron(sx, idb)) * self.Hx
			Ham += sp.kron(ida, sp.kron(sy, idb)) * self.Hy
			Ham += sp.kron(ida, sp.kron(sz, idb)) * self.Hz

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


def TwoSpinOps(PairInd_, PairCoef_, Dof1, Dof2, Nsite):
	Hilsize = Dof1.shape[0]
	Nbonds = len(PairCoef_)  # Number of non-zero bonds
	Hamtmp_ = sp.eye(Hilsize ** Nsite, dtype=complex) * 0
	for i in range(0, Nbonds):
		ia = PairInd_[i, 0]
		ib = PairInd_[i, 1]
		coef = PairCoef_[i]

		ida = sp.eye(Hilsize ** ia)
		idm = sp.eye(Hilsize ** (ib - ia - 1))
		idb = sp.eye(Hilsize ** (Nsite - ib - 1))

		tmp = sp.kron(sp.kron(sp.kron(sp.kron(ida, Dof1), idm), Dof2), idb)
		Hamtmp_ += tmp * coef

	return Hamtmp_


# para = Parameter("../input.inp")
# Lat = Lattice(para)
# Ham = Hamiltonian(Lat)
