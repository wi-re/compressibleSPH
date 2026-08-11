from ..configurations.region import RegionType
from ..geometry import ParticleSet

def filterRegion(region, regions):
    particles = region.particles
    if region.type == RegionType.Boundary:
        return region
    for region_ in regions:
        if region_.type == RegionType.Boundary:
            sdfValues, sdfNormals = region_.sdf(particles.positions)
            mask = sdfValues > 0
            particles = ParticleSet(
                positions = particles.positions[mask],
                supports = particles.supports[mask],
                masses = particles.masses[mask],
                densities = particles.densities[mask]            
            )
            region.particles = particles
    return region

