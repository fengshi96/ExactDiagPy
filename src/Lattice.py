import numpy as np
import math as m
from src.Parameter import Parameter

class Lattice:
	
	def __init__(self, para):
		
		self.LLX = para.LLX
		self.LLY = para.LLY
		self.IsPeriodicX = para.IsPeriodicX
		self.IsPeriodicY = para.IsPeriodicY
		self.Model = para.Model
		self.Nsite = self.LLX * self.LLY * 2
		
		if para.Model=="Kitaev":
			self.Number1neigh = 3 
		elif para.Model=="2dHeisenberg":
			self.Number1neigh = 4
		else:
			pass
		
		self.indx_ = np.zeros(self.Nsite,dtype=int)	
		self.indy_ = np.zeros(self.Nsite,dtype=int)
		self.mesh_ = -np.ones((self.LLX*2+self.LLY, self.LLY*2),dtype=int)	
		self.nn_ = np.zeros((self.Nsite,self.Number1neigh),dtype=int)
		
		self.Build()
		
		
	def Build(self):
		"Construct Lattice mesh and nearest neighbor matrix"
		
		print("[Lattice.py] building Honeycomb lattice...")
		scalex = 2; 
		scaley = 4.0/m.sqrt(3)
		t1 = [1.0 * scalex, 0]
		t2 = [0.5 * scalex, m.sqrt(3)/2.0 * scaley]		
		
		xv = 0
		counter = 0
	
		for i in range(0,self.LLX):
			if i != 0:
				xv += t1[0]
			
			xa = xv; xb = xv + 1
			ya = 0; yb = 1
		
			for j in range(0,self.LLY):
				xa = xv + j*t2[0]; xb = (xv + 1) + j*t2[0]
				ya = j*t2[1]; yb = 1 + j*t2[1]
				
				xa = int(xa); xb = int(xb)
				ya = int(ya); yb = int(yb)			
			
				self.indx_[counter] = xa
				self.indy_[counter] = ya
				self.mesh_[xa,ya] = counter
				counter += 1
				
				self.indx_[counter] = xb
				self.indy_[counter] = yb
				self.mesh_[xb,yb] = counter
				counter += 1	
		print(*self.mesh_,sep = "\n")

		
		print("\n[Lattice.py] Looking for nearest neighbors...")
		xmax = max(self.indx_)
		ymax = max(self.indy_)	
				
		for i in range(0,self.Nsite):
			ix = self.indx_[i]  # coordinate of n-th site in matrix
			iy = self.indy_[i]
			
			#----------------------------OBC-----------------------------------
			
			# n.n in x-bond
			jx = ix + 1; jy = iy + 1  # move 1 step in x = (1,1) direction
			if jx <= xmax and jy <= ymax and self.mesh_[jx,jy] != -1:
				j = self.mesh_[jx,jy]  # site index of n.n. in x direction
				self.nn_[i,0] = j
				self.nn_[j,0] = i 
				
			# n.n in y-bond
			jx = ix + 1; jy = iy - 1  # move 1 step in x = (1,1) direction
			if jx <= xmax and jy <= ymax and jy >= 0 and self.mesh_[jx,jy] != -1:
				j = self.mesh_[jx,jy]  # site index of n.n. in x direction
				self.nn_[i,1] = j
				self.nn_[j,1] = i 
				
			# n.n in z-bond
			jx = ix; jy = iy + 1  # move 1 step in x = (1,1) direction
			if jx <= xmax and jy <= ymax and self.mesh_[jx,jy] != -1:
				j = self.mesh_[jx,jy]  # site index of n.n. in x direction
				self.nn_[i,2] = j
				self.nn_[j,2] = i 		
						
			#----------------------------OBC-----------------------------------	
			
			
			#--------------------------Apply PBC-------------------------------
			
			if self.IsPeriodicY == True:
				# z-bond
				jx = ix - self.LLY 
				jy = 0
				if jx >= 0 and iy == ymax and self.mesh_[jx,jy]!=-1:
					j = self.mesh_[jx,jy]
					self.nn_[i,2] = j
					self.nn_[j,2] = i
			
			if self.IsPeriodicX == True:
				# y-bond
				jx = ix + 2 * self.LLX - 1 
				jy = iy + 1
				if jx <= xmax and iy <= ymax and iy%2 ==0 and self.mesh_[jx,jy]!=-1:
					j = self.mesh_[jx,jy]
					self.nn_[i,1] = j
					self.nn_[j,1] = i			
				
		print(*self.nn_,sep = "\n")


#param = Parameter("../input.inp")
#lat = Lattice(param)



