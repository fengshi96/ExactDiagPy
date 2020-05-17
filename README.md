# Honeycomb-ExactDiag using PRIMME: Python Realization

The main.py takes one argument in terminal: "python main.py input.inp"

The Parameter class mounts all in the input.inp that are to be used to construct lattice and Hamiltonian.
The lattice structure are automatically printed by Lattice class. You can also call them in the code e.g. :
Lat.mesh_ is the lattice defined on a mesh, with vacant sites = -1. (in Heisenberg there is no vacant sites)
Lat.nn_ is the nearest neighbor matrix I sent to you last week.
etc..

the kernel of Hamiltonian construction is in src/Hamiltonian. The connector matrices are called KzzGraph_, KxxGraph_ ..
