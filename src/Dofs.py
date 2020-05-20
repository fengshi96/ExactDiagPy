import numpy as np
import scipy.sparse as sp

class Dofs:
	
	def __init__(self, spin="SpinHalf"):
		if spin == "SpinHalf":

			self.type = "SpinHalf"

			I = np.array([0,1],dtype=int); J = np.array([1,0],dtype=int)
			V = np.array([0.5,0.5],dtype=complex)
			self.Sx = sp.coo_matrix( (V, (I, J)) )
			
			I = np.array([0,1],dtype=int); J = np.array([1,0],dtype=int)
			V = np.array([-0.5j,0.5j],dtype=complex)			
			self.Sy = sp.coo_matrix( (V, (I, J)) )
			
			I = np.array([0,1],dtype=int); J = np.array([0,1],dtype=int)
			V = np.array([0.5,-0.5],dtype=complex)
			self.Sz = sp.coo_matrix( (V, (I, J)) )
			
			self.I = sp.eye(2)
		elif spin == "SpinOne":
			pass
		else:
			raise TypeError("Dof type not yet supported..")
	







#Spins = Dofs("SpinHalf")
#print(Spins.Sz*0+Spins.Sy)





