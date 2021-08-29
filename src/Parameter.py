import re


class Parameter:
    def __init__(self, path):
        self.parameters = {}
        self.GetParameter(path)

    def add_parameter(self, name):
        self.parameters[name] = None

    def add_value(self, name, strVal):
        if strVal.isnumeric():
            val = int(strVal)
        elif re.match(r"[-+]?\d+[.]\d*", strVal) or re.match(r"[-+]?\d*?e[-+]\d+", strVal):
            val = float(strVal)
        else:
            val = strVal
        self.parameters[name] = val


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

            self.add_parameter(name)
            self.add_value(name, var)

        print("Parameters are ....")
        for k, v in zip(self.parameters.keys(), self.parameters.values()):
            print(k, "=", v)
        print("----------------- End of Parameters -----------------\n")







# param = Parameter("../input.inp")
# param.GetParameter("../input.inp")

