class Parameter:
    def __init__(self, path):
        # For spin lattices
        self.LLX = 1
        self.LLY = 1
        self.IsPeriodicX = True
        self.IsPeriodicY = True
        self.Option = None
        self.Model = "Kitaev"
        self.Kxx = 1.0
        self.Kyy = 1.0
        self.Kzz = 1.0
        self.Hx = 0.0
        self.Hy = 0.0
        self.Hz = 0.0
        self.Nstates = 1
        self.SysIndx = None
        # For Bosons
        self.t = None
        self.U = None
        self.mu = None
        self.maxOccupation = None
        self.GetParameter(path)

    def GetParameter(self, path):
        file = open(path, 'r')
        lines = file.readlines()
        file.close()

        for i in range(0, len(lines)):
            if lines[i] != '\n' and '#' not in lines[i]:
                line = lines[i].strip('\n').strip(' ').split("=")
            # print(line)
            name = line[0]
            var = line[1]

            if name == "LLX":
                self.LLX = int(var)
            elif name == "LLY":
                self.LLY = int(var)
            elif name == "IsPeriodicX":
                self.IsPeriodicX = bool(int(var))
            elif name == "IsPeriodicY":
                self.IsPeriodicY = bool(int(var))
            elif name == "Model":
                self.Model = var
            elif name == "Kxx":
                self.Kxx = float(var)
            elif name == "Kyy":
                self.Kyy = float(var)
            elif name == "Kzz":
                self.Kzz = float(var)
            elif name == "Bxx":
                self.Hx = float(var)
            elif name == "Byy":
                self.Hy = float(var)
            elif name == "Bzz":
                self.Hz = float(var)
            elif name == "Nstates":
                self.Nstates = int(var)
            elif name == "Option":
                self.Option = var.strip(' ').split(',')
                if name == "SysIndx":
                    self.SysIndx = eval(var)
                else:
                    ValueError('EE option is on, but no partition of sys and env is specified')
            elif name == "t":
                self.t = float(var)
            elif name == "U":
                self.U = float(var)
            elif name == "mu":
                self.mu = float(var)
            elif name == "maxOccupation":
                self.maxOccupation = int(var)
            else:
                pass

        print("Parameters are ....")
        print("LLX=", self.LLX, "\nLLY=", self.LLY)
        print("IsPeriodicX=", self.IsPeriodicX, "\nIsPeriodicY=", self.IsPeriodicY)
        print("Model=", self.Model)
        if "Hubbard" in self.Model:
            print("t=", self.t)
            print("U=", self.U)
            print("mu=", self.mu)
            print("maxOccupation=", self.maxOccupation)
        else:
            print("Kx=", self.Kxx, "\nKy=", self.Kyy, "\nKz=", self.Kzz)
            print("Hx=", self.Hx, "\nHy=", self.Hy, "\nHz=", self.Hz)
        print("NStates2Keep:", self.Nstates)
        print("option:", self.Option)
        if self.Option is not None:
            if "EE" in self.Option:
                print("System index=", self.SysIndx)
        print("----------------- End of Parameters -----------------\n")

# param = Parameter("../input.inp")
# param.GetParameter("../input.inp")
# print(param.Hx)
