import torch

def domainSDF(p, domain, invert = False):
    center = (domain.min + domain.max) / 2
    halfExtents = (domain.max - domain.min) / 2
    d = torch.abs(p - center) - halfExtents
    return (1 if invert else -1) * (torch.linalg.norm(d.clamp(min = 0), dim = -1) + torch.max(d, dim = -1)[0].clamp(max=0))

def sampleDomainSDF(x, domain, invert = False):
    x_ = x.clone()
    x_.requires_grad = True
    d = domainSDF(x_, domain, invert = invert)
    grad = torch.autograd.grad(outputs = d, inputs = x_, grad_outputs = torch.ones_like(d), create_graph = True, retain_graph = True)

    return d.detach() if x.requires_grad == False else d, grad[0].detach() if x.requires_grad == False else grad[0]
