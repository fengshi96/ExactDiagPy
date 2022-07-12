using HDF5
using LinearAlgebra
using Glob
using Random

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
    return ee, PentS, rdm
end

function OneShotZ!(ρ::Array{Matrix{ComplexF64}}, site::Int, rngIndx=1234)
    Nsites = Int(log2(size(ρ[1])[1]))
    rng = MersenneTwister(rngIndx)
    ida = Matrix{ComplexF64}(I, 2^(site-1), 2^(site-1))
    idb = Matrix{ComplexF64}(I, 2^(Nsites-site), 2^(Nsites-site))
    sz = [1 0 ; 0 -1]
    PauliZ = kron(ida, sz, idb)
    Id = Matrix{ComplexF64}(I, 2^Nsites, 2^Nsites)

    Z = tr(ρ[1] * PauliZ)
    prob_up = real(0.5 * (1 + Z))
   
    if prob_up > rand(rng)
        sample = 1  # up
        ρ[1] = 0.25 * (Id + PauliZ) * ρ[1] * (Id + PauliZ)
        ρ[1] /= tr(ρ[1])
    else
        sample = -1  # down
        ρ[1] = 0.25 * (Id - PauliZ) * ρ[1] * (Id - PauliZ)
        ρ[1] /= tr(ρ[1])
    end
    return sample
end

function OneShotZ2!(ψ::Array{Vector{ComplexF64}}, site::Int, rngIndx=1234)
    Nsites = Int(log2(size(ψ[1])[1]))
    rng = MersenneTwister(rngIndx)
    ida = Matrix{ComplexF64}(I, 2^(site-1), 2^(site-1))
    idb = Matrix{ComplexF64}(I, 2^(Nsites-site), 2^(Nsites-site))
    sz = [1 0 ; 0 -1]
    PauliZ = kron(ida, sz, idb)
    Id = Matrix{ComplexF64}(I, 2^Nsites, 2^Nsites)

    Z = adjoint(ψ[1]) * PauliZ * ψ[1]
    prob_up = real(0.5 * (1 + Z))
   
    if prob_up > rand(rng)
        sample = 1  # up
        ψ[1] = 0.5 * (Id + PauliZ) * ψ[1]
        ψ[1] /= sqrt(prob_up)
    else
        sample = -1  # down
        ψ[1] = 0.5 * (Id - PauliZ) * ψ[1]
        ψ[1] /= sqrt(1 - prob_up)
    end
    return sample
end

let
# Parameters
sysIndxA = Int8[1,2,6,7] .+ 1
Nsite = 20

# read data
hd5 = h5open("../dataSpec.hdf5","r")
dset=hd5["3.Eigen/Wavefunctions"]
evec=read(dset)
close(hd5)

ψ = evec[1, :]
# ρ = wf_as_mat * adjoint(wf_as_mat)
_,_,ρ = Svn(sysIndxA, Nsite, evec[1,:])
numDof = 4
numShots = 50
samples = Matrix{Int8}(undef, (numShots, numDof))
for i in 1:numShots
    rdm = [ψ] # [ρ]
    tmp = Array{Int}(undef, 0)
    for s in 1:numDof
        push!(tmp, OneShotZ2!(rdm, s, rand(100:9000)))
    end
    println(tmp)
end
# show(stdout, "text/plain", samples); println()
# 
# io = open("SampleTC4.dat", "w")
# for i in 1 : numShots	
#     for j in 1 : 3
#         write(io, string(samples[i, j]) * " ")
# 	end
# 	write(io, "\n")
# end
# close(io)
# 

end

