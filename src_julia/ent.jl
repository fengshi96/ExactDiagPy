using HDF5
using LinearAlgebra

function basisPartition(wf, sysIndx, 
        envIndx, nSites)

    nSys = size(sysIndx)[1]
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
