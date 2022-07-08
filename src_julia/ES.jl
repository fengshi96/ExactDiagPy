using HDF5
using LinearAlgebra
using Glob

function basisPartition(wf, sysIndx, nSites)
    """
    	wf: wavefunction mounted from hdf5
    	sysIndx: Array{Int, 1}, Index of sites for RDM
    	nSites: Int, number of sites of the whole system
    """
    nSys = size(sysIndx)[1]
    envIndx = [i for i = 1:nSites]
    for i = 1:nSys
        filter!(e->e≠sysIndx[i], envIndx)
    end
    nEnv = size(envIndx)[1]
    
    @assert nSys + nEnv == nSites
    
    dimDof = 2
    dimHilSys = dimDof^nSys
    dimHilEnv = dimDof^nEnv

    sysBitConfig = zeros(Bool, (dimHilSys, nSites))
    envBitConfig = zeros(Bool, (dimHilEnv, nSites))

    axisScalar = ones(UInt32, nSites)  # UInt32: upto 2^32 (32 sites ED)
    for i = 1 : nSites
        axisScalar[nSites + 1 - i] *= dimDof^(i - 1)
    end

    for i = 1 : dimDof^nSys
        upperpad::Int = size(digits(dimDof^(nSys - 1), base = 2))[1]
        exBits = digits(i-1, base = 2, pad = upperpad)  # expose bits' position for sys dofs

        for j = 1 : size(sysIndx)[1]
            sysBitConfig[i, sysIndx[j]] = exBits[j]
        end

    end

    for i = 1 : dimDof^nEnv
        upperpad::Int = size(digits(dimDof^(nEnv - 1), base = 2))[1]
        exBits = digits(i-1, base = 2, pad = upperpad)  # expose bits' position for evn dofs

        for j = 1 : size(envIndx)[1]
            envBitConfig[i, envIndx[j]] = exBits[j]
        end

    end


    @assert dimHilSys == size(sysBitConfig)[1]
    @assert dimHilEnv == size(envBitConfig)[1]


    # Index matrix
    matIndx = zeros(UInt32, (dimHilSys, dimHilEnv))    # UInt32: upto 2^32 (32 sites ED)

    for i = 1 : dimHilSys
        for j = 1 : dimHilEnv
            matIndx[i,j] = sum((sysBitConfig[i,:] + envBitConfig[j,:]).* axisScalar) + 1 # +1 is because 1-based
        end
    end
    
    # wavefunction as matrix
    wf_as_mat = zeros(Complex{Float64}, (dimHilSys, dimHilEnv))
    
    for i = 1 : dimHilSys
        for j = 1 : dimHilEnv
            wf_as_mat[i,j] = wf[matIndx[i,j]]
        end
    end
    
    return wf_as_mat
end

function Svn(sysIndx::Vector{Int}, Nsite::Int, wf::Vector{ComplexF64})
    @time wf_as_mat = basisPartition(wf, sysIndx, Nsite)
    rdm = wf_as_mat * adjoint(wf_as_mat)
    entS = eigvals(rdm)
    topop =  count(i->(i<0), entS)
    PentS = entS[topop+1:end]
    ee = - transpose(PentS) * log.(PentS)
    return ee, PentS 
end

# Parameters
sysIndxA = Int8[1] .+ 1 # cut a z-bond + 2 y-bond
sysIndxB = Int8[2,3,7,8] .+ 1 # cut a x-bond + 2 y-bond
sysIndxC = Int8[6] .+ 1 # cut a x-bond + 2 y-bond
sysIndxD = Int8[0,4,5,9,10,11,12,13,14] .+ 1 # cut a x-bond + 2 y-bond
sysIndxAB = Int8[1,2,3,7,8] .+ 1 # cut a x-bond + 2 y-bond
sysIndxBC = Int8[2,3,6,7,8] .+ 1
sysIndxAC = Int8[1,6] .+ 1

Nsite = 15
dirList =  glob("Kzz_*") # ["B_0.20"]

Bs = zeros(Float64, (size(dirList)[1], 1))
ESpecA = Vector{Vector{Float64}}()
ESpecB = similar(ESpecA)
ESpecC = similar(ESpecA)
ESpecD = similar(ESpecA)
ESpecAB = similar(ESpecA)
ESpecBC = similar(ESpecA)
ESpecAC = similar(ESpecA)


# read data
hd5 = h5open("../dataSpec.hdf5","r")
dset=hd5["3.Eigen/Wavefunctions"]
evec=read(dset)
close(hd5)


eeA, PentS_A = Svn(sysIndxA, Nsite, evec[1,:])
push!(ESpecA, PentS_A)
println(eeA)

eeB, PentS_B = Svn(sysIndxB, Nsite, evec[1,:])
push!(ESpecB, PentS_B)
println(eeB)

eeC, PentS_C = Svn(sysIndxC, Nsite, evec[1,:])
push!(ESpecC, PentS_C)
println(eeC)

eeD, PentS_D = Svn(sysIndxD, Nsite, evec[1,:])
push!(ESpecD, PentS_D)
println(eeD)

eeAB, PentS_AB = Svn(sysIndxAB, Nsite, evec[1,:])
push!(ESpecAB, PentS_AB)
println(eeAB)

eeBC, PentS_BC = Svn(sysIndxBC, Nsite, evec[1,:])
push!(ESpecBC, PentS_BC)
println(eeBC)

eeAC, PentS_AC = Svn(sysIndxAC, Nsite, evec[1,:])
push!(ESpecAC, PentS_AC)
println(eeAC)

println(eeA + eeB + eeB - eeAB - eeAC - eeBC + eeD)

#io = open("ESpectrum_8.dat", "w")
#for i in 1 : size(dirList)[1]	
#	write(io, string(Bs[i,1]) * " ")
#    for j in 1 : size(ESpecF[i])[1]
#        write(io, string(ESpecF[i][j]) * " ")
#	end
#	write(io, "\n")
#end
#close(io)
