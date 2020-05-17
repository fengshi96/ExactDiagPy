import numpy as np
import scipy.sparse as sp
from src.Parameter import Parameter
from src.Lattice import Lattice
from src.Dofs import Dofs
from src.Helper import matprint


class Hamiltonian:

	def __init__(self, para):

		self.Model = para.Model
		self.Nsite = para.LLX * para.LLY * 2
		self.Kxx = para.Kxx
		self.Kyy = para.Kyy
		self.Kzz = para.Kzz
		self.Hx = para.Hx
		self.Hy = para.Hy
		self.Hz = para.Hz
		self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
		self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)
		self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=int)

		self.KxxPair_ = np.zeros(())  # pairwise non-zero coupling \\
		self.KyyPair_ = np.zeros(())  # 1st and 2nd cols are site indices
		self.KzzPair_ = np.zeros(())

		self.Kxxcoef_ = np.zeros(())  # pairwise non-zero coupling strength
		self.Kyycoef_ = np.zeros(())
		self.Kzzcoef_ = np.zeros(())

		if self.Model == "Kitaev":
			self.Ham = self.BuildKitaev(para)
		elif self.Model == "Heisenberg":
			self.Ham = self.BuildHeisenberg(para)
		elif self.Model == "Hubbard":
			self.Ham = self.BuildHubbard(para)
		else:
			pass

	def BuildKitaev(self, para):

		lat = Lattice(para)
		nn_ = lat.nn_

		for i in range(0, self.Nsite):
			# Kxx_Conn
			j = nn_[i, 0]
			if i < j and j >= 0:
				self.KxxGraph_[i, j] = self.Kxx
				self.KxxGraph_[j, i] = self.Kxx

			# Kyy_Conn
			j = nn_[i, 1]
			if i < j and j >= 0:
				self.KyyGraph_[i, j] = self.Kyy
				self.KyyGraph_[j, i] = self.Kyy

			# Kzz_Conn
			j = nn_[i, 2]
			if i < j and j >= 0:
				self.KzzGraph_[i, j] = self.Kzz
				self.KzzGraph_[j, i] = self.Kzz

		# print("\nKxxGraph_:", *self.KxxGraph_,sep="\n")
		# print("\nKyyGraph_:", *self.KyyGraph_,sep="\n")
		# print("\nKzzGraph_:", *self.KzzGraph_,sep="\n")
		matprint(self.KxxGraph_)
		matprint(self.KyyGraph_)
		matprint(self.KzzGraph_)

		# only need the upper half of KnnGraph_
		xbonds = int(np.count_nonzero(self.KxxGraph_) / 2)
		ybonds = int(np.count_nonzero(self.KyyGraph_) / 2)
		zbonds = int(np.count_nonzero(self.KzzGraph_) / 2)

		self.KxxPair_ = np.resize(self.KxxPair_, (xbonds, 2))
		self.KyyPair_ = np.resize(self.KyyPair_, (ybonds, 2))
		self.KzzPair_ = np.resize(self.KzzPair_, (zbonds, 2))
		self.Kxxcoef_ = np.resize(self.Kxxcoef_, xbonds)
		self.Kyycoef_ = np.resize(self.Kyycoef_, ybonds)
		self.Kzzcoef_ = np.resize(self.Kzzcoef_, zbonds)

		# extract non-zero x-coupling pairs
		counter = 0
		for i in range(0, self.Nsite):
			for j in range(i, self.Nsite):
				if self.KxxGraph_[i, j] != 0:
					self.KxxPair_[counter, 0] = i
					self.KxxPair_[counter, 1] = j
					self.Kxxcoef_[counter] = self.KxxGraph_[i, j]
					counter += 1

		# extract non-zero y-coupling pairs
		counter = 0
		for i in range(0, self.Nsite):
			for j in range(i, self.Nsite):
				if self.KyyGraph_[i, j] != 0:
					self.KyyPair_[counter, 0] = i
					self.KyyPair_[counter, 1] = j
					self.Kyycoef_[counter] = self.KyyGraph_[i, j]
					counter += 1

		# extract non-zero z-coupling pairs
		counter = 0
		for i in range(0, self.Nsite):
			for j in range(i, self.Nsite):
				if self.KzzGraph_[i, j] != 0:
					self.KzzPair_[counter, 0] = i
					self.KzzPair_[counter, 1] = j
					self.Kzzcoef_[counter] = self.KzzGraph_[i, j]
					counter += 1

		# ---------------------Build Hamiltonian as Sparse Matrix-------------------

		print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")
		Spins = Dofs("SpinHalf")
		Sx = Spins.Sx
		Sy = Spins.Sy
		Sz = Spins.Sz
		I = Spins.I

		Ham = sp.eye(2 ** (self.Nsite), dtype=complex) * 0
		Hamx = sp.eye(2 ** (self.Nsite), dtype=complex) * 0
		Hamy = sp.eye(2 ** (self.Nsite), dtype=complex) * 0
		Hamz = sp.eye(2 ** (self.Nsite), dtype=complex) * 0

		for i in range(0, xbonds):
			ia = self.KxxPair_[i, 0]
			ib = self.KxxPair_[i, 1]
			coef = self.Kxxcoef_[i]

			ida = sp.eye(2 ** (ia))
			idm = sp.eye(2 ** (ib - ia - 1))
			idb = sp.eye(2 ** (self.Nsite - ib - 1))
			tmp1 = sp.kron(ida, Sx)
			tmp2 = sp.kron(tmp1, idm)
			tmp3 = sp.kron(tmp2, Sx)
			tmp4 = sp.kron(tmp3, idb)
			Hamx += tmp4 * coef

		for i in range(0, ybonds):
			ia = self.KyyPair_[i, 0]
			ib = self.KyyPair_[i, 1]
			coef = self.Kyycoef_[i]

			ida = sp.eye(2 ** (ia))
			idm = sp.eye(2 ** (ib - ia - 1))
			idb = sp.eye(2 ** (self.Nsite - ib - 1))
			tmp1 = sp.kron(ida, Sy)
			tmp2 = sp.kron(tmp1, idm)
			tmp3 = sp.kron(tmp2, Sy)
			tmp4 = sp.kron(tmp3, idb)
			Hamy += tmp4 * coef

		for i in range(0, zbonds):
			ia = self.KzzPair_[i, 0]
			ib = self.KzzPair_[i, 1]
			coef = self.Kzzcoef_[i]

			ida = sp.eye(2 ** (ia))
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
			ida = sp.eye(2 ** (i))
			idb = sp.eye(2 ** (self.Nsite - i - 1))
			Ham += sp.kron(ida, sp.kron(Sx, idb)) * self.Hx
			Ham += sp.kron(ida, sp.kron(Sy, idb)) * self.Hy
			Ham += sp.kron(ida, sp.kron(Sz, idb)) * self.Hz

		return Ham

	def BuildHeisenberg(self, para):
		pass

	def BuildHubbard(self, para):
		pass
