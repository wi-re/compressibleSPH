"""Legacy generation utilities.

This module used to contain many hardcoded source/boundary case generators.
Those setups are now represented by TOML casefiles and parsed via
:mod:`warpSPH.caseUtils.waveEquation.casefile` +
:mod:`warpSPH.caseUtils.waveEquation.shape_generation`.

Only common initialization helpers that are still used by runtime code remain here.
"""

from typing import Tuple

import torch

from .damping import DampingProfiles, createDampingProfile
from .shape_generation import sample_box


def generateDomainBox(
    particleState,
    config,
    boxHalfExtents: Tuple[float, float] = (0.5, 0.5),
):
    """Generate a centered axis-aligned domain box mask."""
    return sample_box(
        particleState,
        config,
        half_extents=boxHalfExtents,
        offset=(0.0, 0.0),
        pre_rotation=0.0,
        post_rotation=0.0,
    )


def genInitial(
    particleState,
    config,
    domainBox: bool = True,
    domainDamping: bool = True,
):
    """Create base u/v/c/damping grids and source-id grids.

    - `uSourceGrid` uses positive ids for sources.
    - `cSourceGrid` uses `-1` for boundary walls and positive ids for obstacles.
    """
    device = particleState.positions.device
    n = particleState.positions.shape[0]
    u = torch.zeros(n, device=device)
    v = torch.zeros(n, device=device)
    cGrid = torch.ones(n, device=device)
    dampGrid = torch.zeros(n, device=device)

    uSourceGrid = torch.zeros(n, device=device, dtype=torch.long)
    cSourceGrid = torch.zeros(n, device=device, dtype=torch.long)

    if domainBox:
        box = generateDomainBox(
            particleState,
            config,
            boxHalfExtents=(0.95, 0.95),
        )
        cGrid = torch.where(box > 0, cGrid, 0.01)
        cSourceGrid = torch.where(box > 0, cSourceGrid, -1)

    if domainDamping:
        dampGrid = createDampingProfile(
            particleState,
            config,
            DampingProfiles.borderDamping_strong,
        )

    return u, v, cGrid, dampGrid, uSourceGrid, cSourceGrid


__all__ = ["generateDomainBox", "genInitial"]
