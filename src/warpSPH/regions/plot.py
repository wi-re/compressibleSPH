"""Debug/diagnostic plotting of a scene's regions onto a matplotlib axis:
each region's contour as a coloured line (colour/style keyed off
`RegionType`/`BoundaryConditionType`) and, optionally, its particles as a
scatter.
"""

from ..configurations.region import RegionType, ParticleRegion, BoundaryCondition, BoundaryConditionType

__all__ = ['plotRegions']


def plotRegions(regions, axis, plotFluid = True, plotParticles = True):
    for region in regions:
        # visualizeParticles(region['particles'], axis[0,0], config)
        for ic, contour in enumerate(region.contour):
            color = 'black'
            style = '-'
            if isinstance(region, ParticleRegion):
                if region.type == RegionType.Inlet:
                    color = 'green'
                    style = '--'
                if region.type == RegionType.Outlet:
                    color = 'red'
                    style = ':'
                if region.type == RegionType.Boundary:
                    color = 'grey'
                    style = '--'
                if region.type == RegionType.Fluid:
                    color = 'purple'
                    style = '--'
                if region.type == RegionType.Fluid and not plotFluid:
                    continue
            elif isinstance(region, BoundaryCondition):
                if region.type == BoundaryConditionType.constant:
                    color = 'blue'
                    style = ':'
                if region.type == BoundaryConditionType.reflective:
                    color = 'black'
                    style = ':'
                if region.type == BoundaryConditionType.extending:
                    color = 'orange'
                    style = '--'
                if region.type == BoundaryConditionType.forcing:
                    color = 'black'
                    style = '-'
            # axis[0,0].plot(contour[:,0], contour[:,1], color=color)
            axis.plot(contour[:,0], contour[:,1], color = color, ls = style, label = region.type if ic == 0 else None)
        if plotParticles:
            if region.type == RegionType.Inlet and plotFluid: 
                axis.scatter(region.particles.positions[:,0].detach().cpu().numpy(), region.particles.positions[:,1].detach().cpu().numpy(), color = 'green', s = 1)
            if region.type == RegionType.Fluid and plotFluid: 
                axis.scatter(region.particles.positions[:,0].detach().cpu().numpy(), region.particles.positions[:,1].detach().cpu().numpy(), color = 'purple', s = 1)
            if region.type == RegionType.Boundary:
                axis.scatter(region.particles.positions[:,0].detach().cpu().numpy(), region.particles.positions[:,1].detach().cpu().numpy(), color = 'grey', s = 1)
    # axis[0,0].legend()
    