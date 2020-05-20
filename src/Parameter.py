class Parameter:
	def __init__(self,path):
		self.LLX = 1
		self.LLY = 1
		self.IsPeriodicX = True
		self.IsPeriodicY = True
		self.Model = "Kitaev"
		self.Kxx = 1.0
		self.Kyy = 1.0
		self.Kzz = 1.0
		self.Hx = 0.0
		self.Hy = 0.0
		self.Hz = 0.0
		self.Nstates = 1
		self.GetParameter(path)
	
	def GetParameter(self,path):
		file = open(path, 'r')		
		lines = file.readlines()
		file.close()
		
		for i in range(0,len(lines)):
			if lines[i] != '\n' and '#' not in lines[i]:
				line = lines[i].strip('\n').split("=")
			#print(line)
			name = line[0]
			var = line[1]
			
			if name=="LLX":
				self.LLX = int(var)
			elif name=="LLY":
				self.LLY = int(var)
			elif name=="IsPeriodicX":
				self.IsPeriodicX = bool(var)
			elif name=="IsPeriodicY":
				self.IsPeriodicY = bool(var)
			elif name=="Model":
				self.Model = var
			elif name=="Kxx":
				self.Kxx = float(var)
			elif name=="Kyy":
				self.Kyy = float(var)
			elif name=="Kzz":
				self.Kzz = float(var)	
			elif name=="Bxx":
				self.Hx = float(var)	
			elif name=="Byy":
				self.Hy = float(var)
			elif name=="Bzz":
				self.Hz = float(var)	
			elif name=="Nstates":
				self.Nstates = int(var)
			else:
				pass
				
		print("Parameters:")
		print("LLX=", self.LLX, "\nLLY=",self.LLY)
		print("IsPeriodicX=", self.IsPeriodicX,"\nIsPeriodicY=", self.IsPeriodicY)
		print("Model=", self.Model)
		print("Kx=", self.Kxx, "\nKy=", self.Kyy, "\nKz=", self.Kzz)
		print("Hx=", self.Hx, "\nHy=", self.Hy, "\nHz=", self.Hz)
		print("#States2Keep:", self.Nstates)
		
		print("\n-----------------")
				
#param = Parameter("../input.inp")
#param.GetParameter("../input.inp")	
#print(param.Hx)				
			
	
	
	
	
	
	
	
	
	
