#!/bin/sh -x

# Your path to ED_Python
EDPY="/Barn/Lab/ED_Python"

# Parameters for TIFM
Kxx=0.0
Kyy=0.0
Kzz=-1.0
Byy=0.0
Bzz=0.0
SysIndx="[0,1,2]"
Ntates=2

# Lattice geometry
LLX=8
LLY=1
PBCX=1
PBCY=0

# Range of Bxx loop
Bxxmin=0.00
BxxMax=2.00
Step=0.05

#########################################################################
########################    Loop for Bxx    #############################
#########################################################################
mkdir -p 0.ESCollector
for Bxx in $(seq $Bxxmin $Step $BxxMax)
do
  mkdir -p Bxx_n$Bxx
  cd Bxx_n$Bxx || exit
  cp ../input_template.inp ./input.inp

  sed -i 's/#LLXLLX/'$LLX'/g' input.inp
  sed -i 's/#LLYLLY/'$LLY'/g' input.inp
  sed -i 's/#PBCX/'$PBCX'/g' input.inp
  sed -i 's/#PBCY/'$PBCY'/g' input.inp

  sed -i 's/#KXKXKX/'$Kxx'/g' input.inp
  sed -i 's/#KYKYKY/'$Kyy'/g' input.inp
  sed -i 's/#KZKZKZ/'$Kzz'/g' input.inp
  sed -i 's/#BXBXBX/'-$Bxx'/g' input.inp
  sed -i 's/#BYBYBY/'$Byy'/g' input.inp
  sed -i 's/#BZBZBZ/'$Bzz'/g' input.inp 
  sed -i 's/#SYSTEMINDEX/'$SysIndx'/g' input.inp  
  sed -i 's/#NSTATESNSTATES/'$Ntates'/g' input.inp 
  
  python $EDPY/main.py input.inp &>/dev/null
  

  echo "-------Bxx=-$Bxx is done-------"
  cp entspec.dat ../0.ESCollector/entspec_n$Bxx.dat
  cd ..


done












