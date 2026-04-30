
import torch


def getPeriodicPositions(x, domain):
    minD = domain.min.detach().to(x.device)
    maxD = domain.max.detach().to(x.device)
    periodicity = domain.periodic
    pos = [(torch.remainder(x[:, i] - minD[i], maxD[i] - minD[i]) + minD[i]) if periodicity[i] else x[:,i] for i in range(domain.dim)]
    modPos = torch.stack(pos, dim = -1)
    return modPos
