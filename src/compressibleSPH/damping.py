from enum import Enum
import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import os
import torch
if torch.cuda.is_available():
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

import torch

from .utils.sdf import operatorDict, getSDF
from .utils.math import getPeriodicPositions


def sampleDamping(particleState, config, dampingWidth=0.2, dampingStrength=5.0, profile='cosine', periodic_mode=False, global_damping=0.0):
    """
    Create an absorbing boundary layer or global damping.
    
    Parameters:
    -----------
    dampingWidth : float
        Width of the damping layer as a fraction of domain size (0 to 1)
    dampingStrength : float
        Maximum damping coefficient in the boundary layer
    profile : str
        'cosine', 'polynomial', or 'exponential'
    periodic_mode : bool
        If True, use uniform global damping instead of edge damping
        (appropriate for periodic domains)
    global_damping : float
        Uniform damping coefficient applied everywhere (for periodic domains)
        Typical values: 0.01-0.1 for gentle dissipation
    """
    positions = getPeriodicPositions(particleState.positions, config.domain).cpu()
    dampGrid = torch.zeros(positions.shape[0], device=positions.device)
    
    if periodic_mode:
        # For periodic domains: apply uniform weak damping everywhere
        # This gradually dissipates energy without creating artificial boundaries
        dampGrid[:] = global_damping
        return dampGrid.to(particleState.positions.device)
    
    # Non-periodic mode: traditional edge damping
    l = 1.0 - dampingWidth  # Inner boundary of damping layer
    
    # Calculate distance from domain center
    dist = torch.sqrt(positions[:,0]**2 + positions[:,1]**2)
    dist_max = torch.sqrt(torch.tensor(2.0))  # Maximum distance in domain
    
    # Also consider rectangular distance for better corner coverage
    rect_dist = torch.maximum(torch.abs(positions[:,0]), torch.abs(positions[:,1]))
    
    if profile == 'cosine':
        # Smooth cosine taper - works well for absorbing boundaries
        mask = rect_dist > l
        normalized_dist = (rect_dist[mask] - l) / dampingWidth
        dampGrid[mask] = dampingStrength * (1 - torch.cos(normalized_dist * torch.pi / 2))
    
    elif profile == 'polynomial':
        # Cubic polynomial - smoother than exponential
        mask = rect_dist > l
        normalized_dist = (rect_dist[mask] - l) / dampingWidth
        dampGrid[mask] = dampingStrength * normalized_dist**3
    
    elif profile == 'exponential':
        # Original exponential profile but smoother
        sphere_b = lambda points: getSDF('box')['function'](points, torch.tensor([l, l]))
        sdf = operatorDict['invert'](sphere_b)
        dampGrid = sdf(positions).to('cuda')
        
        mask = dampGrid < 0
        dampGrid[mask] = dampingStrength * (1 - torch.exp(-((-dampGrid[mask]) / dampingWidth)**2))
        dampGrid[~mask] = 0
        
        return dampGrid.to(particleState.positions.device)
    
    return dampGrid.to(particleState.positions.device)

def spectral_damping_weights(nx, dim, k_cutoff_fraction=0.5, power=4):
    """
    Create spectral damping weights for high-frequency filtering in periodic domains.
    
    Parameters:
    -----------
    nx : int
        Number of particles per dimension
    dim : int
        Spatial dimension
    k_cutoff_fraction : float
        Fraction of max wavenumber where damping starts (0-1)
    power : int
        Power for damping ramp (higher = sharper cutoff)
    
    Returns:
    --------
    weights : torch.Tensor
        Damping weights in spectral space (1 = no damping, 0 = full damping)
    """
    # Create wavenumber grid
    k = torch.fft.fftfreq(nx, d=1.0/nx)
    
    if dim == 2:
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        k_mag = torch.sqrt(kx**2 + ky**2)
    elif dim == 1:
        k_mag = torch.abs(k)
    else:
        raise NotImplementedError("Only 1D and 2D supported")
    
    k_max = k.abs().max()
    k_cutoff = k_cutoff_fraction * k_max
    
    # Create smooth damping profile in spectral space
    weights = torch.ones_like(k_mag)
    mask = k_mag > k_cutoff
    normalized_k = (k_mag[mask] - k_cutoff) / (k_max - k_cutoff)
    weights[mask] = torch.exp(-(normalized_k**power))
    
    return weights


def apply_spectral_filter(field, nx, dim, k_cutoff_fraction=0.5, power=4):
    """
    Apply spectral filtering to dampen high frequencies in a periodic domain.
    
    This is more appropriate than spatial damping for truly periodic domains.
    """
    # Reshape to grid
    field_grid = field.view(nx, nx) if dim == 2 else field
    
    # FFT to spectral space
    field_fft = torch.fft.fftn(field_grid)
    
    # Get damping weights
    weights = spectral_damping_weights(nx, dim, k_cutoff_fraction, power)
    weights = weights.to(field_fft.device)
    
    # Apply filter
    field_fft_filtered = field_fft * weights
    
    # Back to physical space
    field_filtered = torch.fft.ifftn(field_fft_filtered).real
    
    return field_filtered.flatten()

# Uncomment to visualize:
# compare_damping_profiles(particleState, config)

class DampingProfiles(Enum):
    noDamping = 0
    globalDamping = 1
    globalDamping_strong = 2
    borderDamping = 3
    borderDamping_strong = 4
    randomDamping = 5

def createDampingProfile(particleState, config, profile: DampingProfiles):
    # profile = DampingProfiles.borderDamping_strong
    if profile == DampingProfiles.noDamping:
        dampGrid = torch.zeros(particleState.positions.shape[0], device=particleState.positions.device)
    elif profile == DampingProfiles.globalDamping:
        dampGrid = sampleDamping(particleState, config, 
                                periodic_mode=True,     
                                global_damping=0.05)
    elif profile == DampingProfiles.globalDamping_strong:
        dampGrid = sampleDamping(particleState, config, 
                                periodic_mode=True,     
                                global_damping=0.1)
    elif profile == DampingProfiles.borderDamping:
        dampGrid = sampleDamping(particleState, config, 
                                dampingWidth=0.25, 
                                dampingStrength=15.0, 
                                profile='exponential')
    elif profile == DampingProfiles.borderDamping_strong:
        dampGrid = sampleDamping(particleState, config, 
                                dampingWidth=0.25, 
                                dampingStrength=48.0, 
                                profile='cosine')
    elif profile == DampingProfiles.randomDamping:
        dampGrid = 0.5 * torch.rand(particleState.positions.shape[0], device=particleState.positions.device)
    return dampGrid
