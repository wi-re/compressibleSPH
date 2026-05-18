import math
from sphWarpCore.kernels.wp_kernel import eval_k, eval_dkdq, eval_C_d, iPow
import warp as wp

@wp.kernel
def computePsi(
    kernel: wp.int32,
    dim: wp.int32,
    n_min: wp.float32,
    n_max: wp.float32,
    nLUT: wp.int32,
    n_out: wp.array(dtype=wp.float32, ndim=1), # type: ignore
    psi: wp.array(dtype=wp.float32, ndim=1), # type: ignore
    psiH: wp.array(dtype=wp.float32, ndim=1), # type: ignore
    N_H: wp.array(dtype=wp.float32, ndim=1) # type: ignore
    
):
    i = wp.tid()
    if i >= nLUT:
        return
    
    dn = (n_max - n_min) / (wp.float32(nLUT) - 1.0)
    n_h = n_min + wp.float32(i) * dn

    n_out[i] = n_h

    h = wp.float32(2.0)
    spacing = h / n_h
    dxx = spacing
    v = iPow(dxx, dim)
    x_range = wp.int32(20)
    y_range = wp.int32(20) if dim >= 2 else wp.int32(0)
    z_range = wp.int32(20) if dim >= 3 else wp.int32(0)

    vH = wp.float32(0.0)
    if dim == 1:
        vH = 2.0 * h
    elif dim == 2:
        vH = math.pi * h * h
    elif dim == 3:
        vH = 4.0/3.0 * math.pi * h * h * h

    numNeighbors = wp.int32(0)
    kSum = wp.float32(0.0)
    gradSum = wp.float32(0.0)


    for x in range(-x_range, x_range + 1):
        for y in range(-y_range, y_range + 1):
            for z in range(-z_range, z_range + 1):
                x2 = wp.float32(x) * wp.float32(x) * dxx * dxx
                y2 = wp.float32(y) * wp.float32(y) * dxx * dxx
                z2 = wp.float32(z) * wp.float32(z) * dxx * dxx
                r2 = x2 + y2 + z2
                r = safe_sqrt(r2)
                
                q = abs(r) / h
                W = eval_k(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim)
                gradW = (eval_dkdq(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim + 1))

                if q <= 1.0:
                    numNeighbors += 1

                    kSum += W
                    gradSum += gradW

    deta = 1.0 / n_h
    etar = deta
    eta_max_ = 1.0

    result = wp.float32(0.0)
    while etar < eta_max_:
        correction = wp.float32(0.0)
        if dim == 1:
            correction = 2.0
        elif dim == 2:
            correction = 2.0 * math.pi * etar / deta
        elif dim == 3:
            correction = 4.0 * math.pi * (etar * etar) / (deta * deta)

        q = etar
        val = wp.abs(eval_dkdq(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(eta_max_, dim + 1))
        result += val * correction
        etar += deta

    psiH_0 = result ** (1.0 / wp.float32(dim))

    hReferenceFactor_WH = 1.0 / iPow(2.0, dim)
    hActualFactor_WH = 1.0 / iPow(h, dim)
    hScaling_WH = hReferenceFactor_WH / hActualFactor_WH

    psi_0 = (hScaling_WH * kSum)**wp.float32(1.0/wp.float32(dim))

    psi[i] = psi_0
    psiH[i] = psiH_0
    N_H[i] = vH / v


    
import torch
from sphWarpCore import KernelFunctions
def generatePSILut_warp(
        
    kernel: KernelFunctions,
    n_min: float,
    n_max: float,
    nLut: int
):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    psis = torch.zeros((nLut,3), dtype=torch.float32, device=device)
    psiHs = torch.zeros((nLut,3), dtype=torch.float32, device=device)
    N_Hs = torch.zeros((nLut,3), dtype=torch.float32, device=device)

    for d in range(1, 4):
        # print(f'Generating LUT for dim {d}...')
        n_out_wp = wp.zeros(nLut, dtype=wp.float32)
        psi_wp = wp.zeros(nLut, dtype=wp.float32)
        psiH_wp = wp.zeros(nLut, dtype=wp.float32)
        N_H_wp = wp.zeros(nLut, dtype=wp.float32)

        wp.launch(
            computePsi,
            dim = nLut,
            inputs = [
                kernel,
                d,
                n_min,
                n_max,
                nLut,
                n_out_wp,
                psi_wp,
                psiH_wp,
                N_H_wp
            ]
        )
        psis[:, d-1] = wp.to_torch(psi_wp).clone().to(device)
        psiHs[:, d-1] = wp.to_torch(psiH_wp).clone().to(device)
        N_Hs[:, d-1] = wp.to_torch(N_H_wp).clone().to(device)
    return wp.to_torch(n_out_wp).clone().to(device), psis, psiHs, N_Hs