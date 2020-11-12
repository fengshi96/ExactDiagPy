# ExactDiag by PRIMME: Python Realization

The program is designed to exactly diagonalize several common quantum many-body models. It uses iterative methods developed in the project PRIMME (Python wrapper of Fortran). 



*Another realization with Julia for Kitaev's honeycomb can be found in the repo:*
*https://github.com/fengshi96/Honeycomb-ExactDiag*
*which uses ARPACK instead of PRIMME. However it is significantly slower and consumes a lot more mem.*




The program supports the following models: 
(1) Kitaev's honeycomb model + external field
(2) 2D Heisenberg model + external field
(3) 2D Hubbard model < Comming soon... >

The program is dependent on Primme(https://pypi.org/project/primme/). 
To install Primme by pip:
```
pip install numpy   # if numpy is not installed yet
pip install scipy   # if scipy is not installed yet
pip install future  # if using python 2
pip install h5py    # if HDF5 is not installed
conda install mkl-devel # if using Anaconda Python distribution
pip install primme
```

The main.py takes one argument in terminal: 

```
python main.py input.inp
```

The file input.inp includes all parameters that are needed to construct the Lattice and Hamiltonain:
```
LLX, LLY       # Number of unit cells in X and Y direction
Model          # Name of the model: Kitaev, Heisenberg or Hubbard
IsPeriodicX    # PBC (1) or OBC (0) in X direction
IsPeriodicY    # PBC (1) or OBC (0) in Y direction
Kxx, Kyy, Kzz  # The coupling constant of spins
Bxx, Byy, Bzz  # Magnetic field in x, y, z directions
t, U, mu       # Standard Constants in Hubbard Model <comming soon...>
Nstates        # Number of eigen states to keep
```




The Parameter class defined in src/Parameter mounts all in the input.inp that are to be used to construct lattice and Hamiltonian.

The lattice structure are automatically printed by Lattice class. You can also call them in the code e.g.
(1) Lat.mesh_ is the lattice defined on a mesh, with vacant sites = -1. (in Heisenberg there is no vacant sites)
(2)Lat.nn_ is the nearest neighbor matrix.
etc..

the kernel of Hamiltonian construction is in src/Hamiltonian. The connector matrices are called KzzGraph_, KxxGraph_ ..

