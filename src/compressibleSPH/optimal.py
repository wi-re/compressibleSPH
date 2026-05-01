
# def evaluateOptimalSupport(
#         particles: Union[CompressibleState, WeaklyCompressibleState],
#         kernel_: SPHKernel,
#         neighborhood: NeighborhoodInformation = None,
#         supportScheme: SupportScheme = SupportScheme.Scatter,
#         config: Dict = {}
#         ):
#     kernel = getSPHKernelv2(kernel_)
#     rhos = [particles.densities]
#     hs = [particles.supports]
#     neighborhood_ = neighborhood
#     verletScale = 1.4 if neighborhood is None else neighborhood.verletScale
#     nIter = getSetConfig(config, 'support', 'iterations', 16)
#     adaptiveHThreshold = getSetConfig(config, 'support', 'adaptiveHThreshold', 1e-3)
#     supportMode = 'superSymmetric' if neighborhood is None else neighborhood.mode
#     if supportMode not in ['gather', 'superSymmetric']:
#         supportMode = 'superSymmetric'
#     # verletScale = 1.0 if config is None else (config['neighborhood']['verletScale'] if 'verletScale' in config else 1.0)

#     hMin = particles.supports.min()
#     hMax = particles.supports.max()
#     for i in range(nIter):
#         with record_function(f"[SPH] - Optimal Support - Neighbors"):
#             # print(f'Iteration {i} | neighborhood: {neighborhood_ is not None}')
#             neighborhood, neighbors = evaluateNeighborhood(particles, config['domain'], kernel_, verletScale = verletScale, mode = SupportScheme.SuperSymmetric, priorNeighborhood=neighborhood)
#             numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries

#             # if (numNeighbors.max() > 60):
#             #     print(f'Warning: Number of neighbors is too high: {numNeighbors.max().item()}')
#             #     neighs = neighbors.get('noghost')
#             #     kernelValues = neighs[1]
#             #     print('r_ij:', kernelValues.r_ij.min().item(), kernelValues.r_ij.max().item(), kernelValues.r_ij.mean().item())
#             #     print('x_ij:', kernelValues.x_ij.min().item(), kernelValues.x_ij.max().item(), kernelValues.x_ij.mean().item())

#             #     print('h_ij:', particles.supports.min().item(), particles.supports.max().item(), particles.supports.mean().item())
#             #     print('ddh_W_i:', kernelValues.ddh_W_i.min().item(), kernelValues.ddh_W_i.max().item(), kernelValues.ddh_W_i.mean().item())

#             # if (numNeighbors.min() < 40):
#             #     print('Iteration', i, 'has particles with too few neighbors:')
#             #     mask = numNeighbors < 40
#             #     neighs = neighbors.get('noghost')
                
#             #     indices = torch.arange(particles.positions.shape[0], device=particles.positions.device, dtype=torch.int64)
#             #     indices = indices[mask]
                
#             #     for index in indices:
#             #         print(f'Warning: Particle {index} has too few neighbors: {numNeighbors[index].item()}')
#             #         print(f'Position: {particles.positions[index].cpu().numpy()}')
#             #         print(f'Support: {particles.supports[index].item()}')
#             #         print(f'Density: {particles.densities[index].item()}')
#             #         particle_neighbors = neighs[0].col[neighs[0].row == index]
#             #         print(f'Neighbors: {particle_neighbors.cpu().numpy()}')
#             #         print(f'Neighbor Positions: {particles.positions[particle_neighbors].cpu().numpy()}')
#             #         print(f'Neighbor Supports: {particles.supports[particle_neighbors].cpu().numpy()}')
#             #         print(f'Neighbor Densities: {particles.densities[particle_neighbors].cpu().numpy()}')
#             #         print('---')

#             # neighborhood_, sparseNeighborhood = buildNeighborhood(particles, particles, config['domain'], verletScale= verletScale, mode =supportMode, priorNeighborhood=neighborhood_, verbose = False, neighborhoodAlgorithm = config['neighborhood']['algorithm'])
#             # actualNeighbors = filterNeighborhood(neighborhood_)
#             # actualNeighbors = sparseNeighborhood
#             # actualNeighbors = filterNeighborhoodByKind(particles, actualNeighbors)
#             # actualNeighbors, _ = filterNeighborsGhost(neighborhood_, particles, ghostState)
        
#         if isinstance(particles, Tuple):
#             particles.densities = computeDensity(particles, kernel_, neighbors.get('noghost'), supportScheme, config)
            
#             # particles = particles._replace(densities = computeDensity(particles, particles, domain, kernel, actualNeighbors, 'gather'))
#         else:
#             particles.densities = computeDensity(particles, kernel_, neighbors.get('noghost'), supportScheme, config)#computeDensity(particles, particles, domain, kernel, actualNeighbors, 'gather')
#             # if ghostState is not None:
#                 # particles.densities = LiuLiuFirstOrder(particles.densities, particles, ghostState, domain, kernel)
            
#         h_prev = particles.supports
        
#         F_ = F(h_prev, particles.densities, particles.masses, config['targetNeighbors'], dim = particles.positions.shape[1])
#         # dFdh_ = dFdh(particles, kernel, domain, actualNeighbors)
#         dFdh_ = computeOmega(particles, kernel_, neighbors.get('noghost'), supportScheme, config)
        
#         h_new = h_prev - F_ / (dFdh_ + 1e-6)
#         # if ghostState is not None:
#             # h_new[particles.species != 0] = h_prev[particles.species != 0]

#         h_new = h_new.clamp(min = hMin * 0.25, max = hMax * 4.0)
#         hMin = h_new.min()
#         hMax = h_new.max()

#         h_diff = h_new - h_prev
#         h_ratio = h_new / (h_prev + 1e-6)
        
#         verbose = False
#         checkTensor(F_, particles.positions.dtype, particles.positions.device, 'F_', verbose = verbose)
#         checkTensor(dFdh_, particles.positions.dtype, particles.positions.device, 'dFdh_', verbose = verbose)
#         checkTensor(h_new, particles.positions.dtype, particles.positions.device, 'h_new', verbose = verbose)
#         checkTensor(h_diff, particles.positions.dtype, particles.positions.device, 'h_diff', verbose = verbose)
#         checkTensor(h_ratio, particles.positions.dtype, particles.positions.device, 'h_ratio', verbose = verbose)
#         checkTensor(particles.densities, particles.positions.dtype, particles.positions.device, 'densities', verbose = verbose)
        
#         # print(f'Support Update: {h_diff.min()} | {h_diff.max()} | {h_diff.mean()}')
        
#         # print(f'Iteration: {i} | h_ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
#         # print(f'Densities: {particles.densities.min()} | {particles.densities.max()} | {particles.densities.mean()}')
#         # print(f'Supports: {h_new.min()} | {h_new.max()} | {h_new.mean()}')
#         # print(f'F: {F_.min()} | {F_.max()} | {F_.mean()}')
#         # print(f'dFdh: {dFdh_.min()} | {dFdh_.max()} | {dFdh_.mean()}')
#         # print(f'Number of neighbors: {numNeighbors.min().item()} - {numNeighbors.max().item()} - {numNeighbors.median().item()}')

#         # rhos.append(particles.densities)
#         # hs.append(h_new)
        
#         # actualNeighbors = buildSuperSymmetricNeighborhood(particles, domain, verletScale = 1.0)
#         # rho = computeDensity(particles, particles, domain, kernel, actualNeighbors, 'gather' )
#         # V = particles.masses / rho
#         # h = targetNeighbors / 2 * V
        
#         # print(f'Iteration {i} | Support: {h.min()} | {h.max()} | {h.mean()} | Ratio: {(h / hs[0]).min()} | {(h / hs[0]).max()} | {(h / hs[0]).mean()}')
        
        
#         if isinstance(particles, Tuple):
#             particles = particles._replace(supports = h_new)
#         else:
#             particles.supports = h_new
#             # particles.densities = rho

#         # particles = particles._replace(supports = h, densities = rho)
#         rhos.append(particles.densities)
#         hs.append(particles.supports)
        
#         # print(f'Iteration {i} | h_ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
        
#         if (h_ratio - 1).abs().max() < adaptiveHThreshold:
#             # print('Stopping Early')
#             break

#     neighborhood, neighbors = evaluateNeighborhood(particles, config['domain'], kernel_, verletScale = 1.0, mode = SupportScheme.SuperSymmetric, priorNeighborhood=neighborhood)
#     numNeighbors = coo_to_csr(filterNeighborhoodByKind(particles, neighbors.neighbors, which = 'noghost')).rowEntries

#     neighs = neighbors.get('noghost')
#     kernelValues = neighs[1]
#     # print('r_ij:', kernelValues.r_ij.min().item(), kernelValues.r_ij.max().item(), kernelValues.r_ij.mean().item())
#     # print('x_ij:', kernelValues.x_ij.min().item(), kernelValues.x_ij.max().item(), kernelValues.x_ij.mean().item())

#     # print('h_ij:', particles.supports.min().item(), particles.supports.max().item(), particles.supports.mean().item())
#     # print('ddh_W_i:', kernelValues.ddh_W_i.min().item(), kernelValues.ddh_W_i.max().item(), kernelValues.ddh_W_i.mean().item())

#     # if (numNeighbors.max() > 60):
#     #     print(f'Warning: Number of neighbors is too high: {numNeighbors.max().item()}')
#     #     raise RuntimeError('Number of neighbors is too high, please check your configuration or the kernel used.')

#     # if neighborhood is not None:
#     return rhos[-1], hs[-1], rhos, hs, neighborhood_
#     # return rhos[-1], hs[-1], rhos, hs