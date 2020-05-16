import sys, re, math, random
import numpy as np
import scipy.sparse as sp
import primme
from src.Parameter import Parameter
from src.Lattice import Lattice
from src.Hamiltonian import Hamiltonian



def main(total, cmdargs):
	if total != 2:
			print (" ".join(str(x) for x in cmdargs))
			raise ValueError('Missing arguments')
	inputname = cmdargs[1]
	#---------------------------------------------------------------
	
	para = Parameter(inputname)
	Hamil = Hamiltonian(para)
	ham = Hamil.Ham
	
	Nstates = para.Nstates
	evals, evecs = primme.eigsh(ham, Nstates, tol=1e-6, which='SA')
	print("\nEigen Values:\n", *evals, sep='\n')




if __name__ == '__main__':
	sys.argv ## get the input argument
	total = len(sys.argv)
	cmdargs = sys.argv	
	main(total, cmdargs)





