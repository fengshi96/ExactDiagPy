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

# Parameters
sysIndxA = Int8[0,1,6,7] .+ 1
sysIndxB = Int8[2,3,4,5] .+ 1
sysIndxC = Int8[8,9,10,11] .+ 1

sysIndxAB = Int8[0,1,2,3,4,5,6,7] .+ 1
sysIndxBC = Int8[2,3,4,5,8,9,10,11] .+ 1
sysIndxAC= Int8[0,1,6,7,8,9,10,11] .+ 1
sysIndxABC = Int8[12,13,14,15,16,17] .+ 1  # equivalent to D

Nsite = 18
dirList = glob("Kzz_*")

Stopos = zeros(Float32, size(dirList)[1])
counter = 1
for kzdir in dirList
	# read data
	hd5 = h5open(kzdir * "/dataSpec.hdf5","r")
	dset=hd5["3.Eigen/Wavefunctions"]
	evec=read(dset)
	close(hd5)

	# bipartition of wavefunction
	# A
	@time wf_as_mat_A = basisPartition(evec[1,:], sysIndxA, Nsite)

	#B
	@time wf_as_mat_B = basisPartition(evec[1,:], sysIndxB, Nsite)

	#C
	@time wf_as_mat_C = basisPartition(evec[1,:], sysIndxC, Nsite)

	#AB
	@time wf_as_mat_AB = basisPartition(evec[1,:], sysIndxAB, Nsite)

	#BC
	@time wf_as_mat_BC = basisPartition(evec[1,:], sysIndxBC, Nsite)

	#AC
	@time wf_as_mat_AC = basisPartition(evec[1,:], sysIndxAC, Nsite)

	#ABC
	@time wf_as_mat_ABC = basisPartition(evec[1,:], sysIndxABC, Nsite)

	#@show size(wf_as_mat);

	# then trace out enironment
	rdmA = wf_as_mat_A * adjoint(wf_as_mat_A)
	entS_A = eigvals(rdmA)
	topop_A = count(i->(i<0), entS_A)
	PentS_A = entS_A[topop_A+1:end]  # popped out negative values

	rdmB = wf_as_mat_B * adjoint(wf_as_mat_B)
	entS_B = eigvals(rdmB)
	topop_B = count(i->(i<0), entS_B)
	PentS_B = entS_B[topop_B+1:end]  # popped out negative values

	rdmC = wf_as_mat_C * adjoint(wf_as_mat_C)
	entS_C = eigvals(rdmC)
	topop_C = count(i->(i<0), entS_C)
	PentS_C = entS_C[topop_C+1:end]  # popped out negative values

	rdmAB = wf_as_mat_AB * adjoint(wf_as_mat_AB)
	entS_AB = eigvals(rdmAB)
	topop_AB = count(i->(i<0), entS_AB)
	PentS_AB = entS_AB[topop_AB+1:end]  # popped out negative values

	rdmBC = wf_as_mat_BC * adjoint(wf_as_mat_BC)
	entS_BC = eigvals(rdmBC)
	topop_BC = count(i->(i<0), entS_BC)
	PentS_BC = entS_BC[topop_BC+1:end]  # popped out negative values

	rdmAC = wf_as_mat_AC * adjoint(wf_as_mat_AC)
	entS_AC = eigvals(rdmAC)
	topop_AC = count(i->(i<0), entS_AC)
	PentS_AC = entS_AC[topop_AC+1:end]  # popped out negative values

	rdmABC = wf_as_mat_ABC * adjoint(wf_as_mat_ABC)
	entS_ABC = eigvals(rdmABC)
	topop_ABC = count(i->(i<0), entS_ABC)
	PentS_ABC = entS_ABC[topop_ABC+1:end]  # popped out negative values



	# println("\nEigen Spectrum:"); show(stdout, "text/plain", PentS); println()

	@show eeA = - transpose(PentS_A) * log.(PentS_A)
	@show eeB = - transpose(PentS_B) * log.(PentS_B)
	@show eeC = - transpose(PentS_C) * log.(PentS_C)
	@show eeAB = - transpose(PentS_AB) * log.(PentS_AB)
	@show eeBC = - transpose(PentS_BC) * log.(PentS_BC)
	@show eeAC = - transpose(PentS_AC) * log.(PentS_AC)
	@show eeABC = - transpose(PentS_ABC) * log.(PentS_ABC)

	@show S_topo = eeA + eeB + eeC - eeAB - eeBC - eeAC + eeABC
	Stopos[counter] = S_topo
	global counter = counter + 1
end

io = open("topoEE.dat", "w")
for ee in Stopos
	write(io, string(ee) * "\n")
end
close(io)






