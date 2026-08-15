"""Builds the initial particle state and `SimulationSystem` for the weakly
compressible (and incompressible, via the shared `SimulationState` branch)
solver from a list of sampled regions: concatenates each region's particles
into a combined `WeaklyCompressibleState`/`IncompressibleState`, applies
per-region `initialConditions`, adds boundary ghost particles, and builds any
rigid bodies. `initializeSimulation` is the entry point re-exported (as
`initializeWeaklyCompressibleSimulation`) by `initializers/__init__.py`;
`initializeWeaklyCompressibleState` is an older, narrower variant not called
from anywhere else in the package.
"""

import warnings
import torch
from ..configurations import *
from ..systems import *
from ..rigidBody import *
from ..regions import *


def initializeWeaklyCompressibleState(regions, config, verbose = True):
    if 'fluid' not in config:
        if verbose:
            warnings.warn('No fluid configuration found. Using default values.')
        config['fluid'] = {}
    if 'c_s' not in config['fluid']:
        if verbose:
            warnings.warn('No speed of sound found. Using default value.')
        config['fluid']['c_s'] = 10       
    if 'rho0' not in config['fluid']:
        if verbose:
            warnings.warn('No reference density found. Using default value.')
        config['fluid']['rho0'] = 1

    rho0 = config.get('fluid',{}).get('rho0', 1)
    c_s = config.get('fluid',{}).get('c_s', 1)
    dim = config['domain'].dim
    
    fluidParticles = []
    for region in regions:
        if region['type'] == 'fluid' or region['type'] == 'inlet':
            fluidParticles.append(region)
    if len(fluidParticles) == 0:
        raise ValueError('No fluid particles found. Please check the regions.')

    if 'particle' not in config:
        if verbose:
            warnings.warn('No particle configuration found.')
        config['particle'] = {}

    if 'support' not in config['particle']:
        if verbose:
            warnings.warn('Using default support configuration.')
        config['particle']['support'] = fluidParticles[0]['particles'].supports.mean().item()
    if 'dx' not in config['particle']:
        if verbose:
            warnings.warn('Using default dx configuration.')
        config['particle']['dx'] = fluidParticles[0]['particles'].masses.pow(1/dim).mean().item()

    device = fluidParticles[0]['particles'].positions.device
    dtype = fluidParticles[0]['particles'].positions.dtype

    positions = []
    supports = []
    masses = []
    velocities = []
    densities = []
    pressures = []
    soundspeeds = []
    kinds = []
    materials = []

    particleRegions = []
    for region in regions:
        if region['type'] == 'fluid' or region['type'] == 'boundary':
            particleRegions.append(region)
            positions.append(region['particles'].positions)
            supports.append(getInitialValue(region, region['particles'].positions, 'supports', region['particles'].supports))
            masses.append(getInitialValue(region, region['particles'].positions, 'masses', region['particles'].masses))
            velocities.append(getInitialValue(region, region['particles'].positions, 'velocities', torch.zeros_like(region['particles'].positions)))
            densities.append(getInitialValue(region, region['particles'].positions, 'densities', torch.ones_like(region['particles'].masses) * config['fluid']['rho0']))
            pressures.append(getInitialValue(region, region['particles'].positions, 'pressures', torch.zeros_like(region['particles'].masses)))
            soundspeeds.append(getInitialValue(region, region['particles'].positions, 'soundspeeds', torch.ones_like(region['particles'].masses) * config['fluid']['c_s']))
    fluidMaterials = 0
    boundaryMaterials = 0
    
    for region in particleRegions:
        if region['type'] == 'fluid':
            kinds.append(torch.zeros_like(region['particles'].masses, dtype = torch.int32))
            materials.append(torch.ones_like(region['particles'].masses, dtype = torch.int32) * fluidMaterials)
            fluidMaterials += 1
        elif region['type'] == 'boundary':
            kinds.append(torch.ones_like(region['particles'].masses, dtype = torch.int32))
            materials.append(torch.ones_like(region['particles'].masses, dtype = torch.int32)* boundaryMaterials)
            boundaryMaterials += 1
            
    kinds = torch.cat(kinds, dim = 0)
    materials = torch.cat(materials, dim = 0)
    positions = torch.cat(positions, dim = 0)
    UIDs = torch.arange(positions.shape[0], device = device, dtype = torch.int64)

    supports = torch.cat(supports, dim = 0)
    masses = torch.cat(masses, dim = 0)
    velocities = torch.cat(velocities, dim = 0)
    densities = torch.cat(densities, dim = 0)
    pressures = torch.cat(pressures, dim = 0)
    soundspeeds = torch.cat(soundspeeds, dim = 0)

    particleState = WeaklyCompressibleState(
        positions,
        supports = supports,
        masses = masses * config.get('fluid',{}).get('rho0', 1),
        
        densities = densities,
        velocities = velocities,
        
        pressures = pressures,
        soundspeeds = soundspeeds,
        
        kinds = kinds,
        materials = materials,
        UIDs = UIDs,
        
        UIDcounter = positions.shape[0],
    )

    return particleState, config

from warpSPH.configurations import *

def getInitialValue(region, pos, quantity, default):
    if region.initialConditions is None or quantity not in region.initialConditions:
        return default
    if callable(region.initialConditions[quantity]):
        return region.initialConditions[quantity](pos)
    else:
        temp = torch.zeros_like(default)
        temp[:] = region.initialConditions[quantity]
        return temp

from ..systems.incompressible import *

def initializeState(regions, config, schemeConfig, SimulationState, verbose = True):    
    rho0 = schemeConfig.fluid.restDensity if isinstance(schemeConfig, WeaklyCompressibleSPHConfig) or isinstance(schemeConfig, IncompressibleSPHConfig) else config.rho0
    uidCounter = 0

    states = []
    types = [RegionType.Fluid, RegionType.Boundary]
    for ty in types:
        for ir, region in enumerate([r for r in regions if r.type == ty]):
            positions = region.particles.positions
            
            if SimulationState is WeaklyCompressibleState:
                tempState = WeaklyCompressibleState(
                    positions,
                    supports = getInitialValue(region, positions, 'supports', region.particles.supports),
                    masses = getInitialValue(region, positions, 'masses', region.particles.masses) * rho0,
                    
                    densities = getInitialValue(region, positions, 'densities', torch.ones_like(region.particles.masses) * rho0),
                    velocities = getInitialValue(region, positions, 'velocities', torch.zeros_like(region.particles.positions)),
                    
                    pressures = getInitialValue(region, positions, 'pressures', None),
                    soundspeeds = getInitialValue(region, positions, 'soundspeeds', None),
                    
                    kinds = torch.ones_like(region.particles.masses, dtype = torch.int32) * (0 if region.type == RegionType.Fluid else 1),
                    materials = torch.ones_like(region.particles.masses, dtype = torch.int32) * ir,
                    UIDs = torch.arange(uidCounter, uidCounter + positions.shape[0], device = positions.device, dtype = torch.int64),
                    
                    UIDcounter = uidCounter + positions.shape[0],

                    ghostIndices = torch.ones_like(region.particles.masses, dtype = torch.int32) * (-1),
                    ghostOffsets = region.particles.positions.clone()
                )
            elif SimulationState is IncompressibleState:
                tempState = IncompressibleState(
                    positions,
                    supports = getInitialValue(region, positions, 'supports', region.particles.supports),
                    masses = getInitialValue(region, positions, 'masses', region.particles.masses) * rho0,
                    
                    densities = getInitialValue(region, positions, 'densities', torch.ones_like(region.particles.masses) * rho0),
                    velocities = getInitialValue(region, positions, 'velocities', torch.zeros_like(region.particles.positions)),
                    
                    pressures = getInitialValue(region, positions, 'pressures', None),
                    soundspeeds = getInitialValue(region, positions, 'soundspeeds', None),
                    
                    kinds = torch.ones_like(region.particles.masses, dtype = torch.int32) * (0 if region.type == RegionType.Fluid else 1),
                    materials = torch.ones_like(region.particles.masses, dtype = torch.int32) * ir,
                    UIDs = torch.arange(uidCounter, uidCounter + positions.shape[0], device = positions.device, dtype = torch.int64),
                    
                    UIDcounter = uidCounter + positions.shape[0],

                    ghostIndices = torch.ones_like(region.particles.masses, dtype = torch.int32) * (-1),
                    ghostOffsets = region.particles.positions.clone()
                )

            uidCounter += positions.shape[0]
            states.append(tempState)

    if SimulationState is WeaklyCompressibleState:
        combinedState = WeaklyCompressibleState(
            positions = torch.cat([s.positions for s in states], dim = 0),
            supports = torch.cat([s.supports for s in states], dim = 0),
            masses = torch.cat([s.masses for s in states], dim = 0),
            
            densities = torch.cat([s.densities for s in states], dim = 0),
            velocities = torch.cat([s.velocities for s in states], dim = 0),
            
            pressures = None if states[0].pressures is None else torch.cat([s.pressures for s in states], dim = 0),
            soundspeeds = None if states[0].soundspeeds is None else torch.cat([s.soundspeeds for s in states], dim = 0),
            
            kinds = torch.cat([s.kinds for s in states], dim = 0),
            materials = torch.cat([s.materials for s in states], dim = 0),
            UIDs = torch.cat([s.UIDs for s in states], dim = 0),
            
            UIDcounter = uidCounter,
            ghostIndices = torch.cat([s.ghostIndices for s in states], dim = 0),
            ghostOffsets = torch.cat([s.ghostOffsets for s in states], dim = 0)
        )
    elif SimulationState is IncompressibleState:
        combinedState = IncompressibleState(
            positions = torch.cat([s.positions for s in states], dim = 0),
            supports = torch.cat([s.supports for s in states], dim = 0),
            masses = torch.cat([s.masses for s in states], dim = 0),
            
            densities = torch.cat([s.densities for s in states], dim = 0),
            velocities = torch.cat([s.velocities for s in states], dim = 0),
            
            pressures = None if states[0].pressures is None else torch.cat([s.pressures for s in states], dim = 0),
            soundspeeds = None if states[0].soundspeeds is None else torch.cat([s.soundspeeds for s in states], dim = 0),
            
            kinds = torch.cat([s.kinds for s in states], dim = 0),
            materials = torch.cat([s.materials for s in states], dim = 0),
            UIDs = torch.cat([s.UIDs for s in states], dim = 0),
            
            UIDcounter = uidCounter,
            ghostIndices = torch.cat([s.ghostIndices for s in states], dim = 0),
            ghostOffsets = torch.cat([s.ghostOffsets for s in states], dim = 0)
        )   

    return combinedState



def initializeSimulation(regions, config, schemeConfig, SimulationSystem, SimulationState, verbose = True):
    particleState = initializeState(regions, config, schemeConfig, SimulationState, verbose = verbose)


    particleState = addBoundaryGhostParticles(regions, particleState)
    # particleState, emitted = processInlets(particleState, config, config['domain'])
    # particleState.densities[:] = config['fluid']['rho0']
    # particleState.velocities = forcing(particleState.positions)

    # neighborhood, sparseNeighborhood_ = buildNeighborhood(particleState, particleState, config['domain'], verletScale = config['neighborhood']['verletScale'], mode = 'superSymmetric', priorNeighborhood=None)         
    # ghostNeighbors = filterNeighborhoodByKind(particleState, sparseNeighborhood_, which = 'fluidToGhost')

    rigidBodyIDs = torch.unique(particleState.materials[particleState.kinds == 1]).cpu().numpy()
    # print(rigidBodyIDs)
    rigidBodies = []
    for id in rigidBodyIDs:
        # print('Processing rigid body', id)
        rigidBody = buildRigidBody(particleState, regions, id)
        if(rigidBody is not None):
            rigidBodies.append(rigidBody)

    # rigidBodies[0].angularVelocity = np.pi / 2
    for rigidBody in rigidBodies:
        particleState = updateBodyParticlesWCSPH(particleState, rigidBody)
    config.rigidBodies = rigidBodies
    config.regions = regions
    
    compressibleSystem = SimulationSystem(
        state=particleState, 
        adjacency = None, 
        domain = config.domain
    )
        
    return compressibleSystem#, config, rigidBodies


__all__ = ['initializeSimulation']