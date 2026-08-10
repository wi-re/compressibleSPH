"""Plotter setup for the weakly-compressible cases.

Moved here from ``datagen/weaklyCompressible/plot.py`` so `warpSPH.cases` can
use it; that path is now a re-export shim for the notebooks.
"""

from warpSPHPlotting import *

__all__ = ['buildPlotText', 'setupPlotter', 'updatePlot']

def buildPlotText(runningState, args, simSetup, config, schemeConfig, timing):
    
    caseText = f'{args.caseName}'
    timeText = f't = {runningState.t:.4g}/{args.timeLimit:.4g} | dt = {config.dt:.4g}'
    particleText = f'particles = {len(runningState.state.positions[runningState.state.kinds == 0])} fluid + {len(runningState.state.positions[runningState.state.kinds == 1])} boundary | nx = {simSetup.nx} | n_h = {simSetup.n_h}'
    domainText = f'L = {simSetup.L}, W = {simSetup.W}'
    obstacleText = f'obstacle: {args.obstacleType}, aoa: {args.aoa}' if args.obstacleActive else 'no obstacle'
    stateText = f'v_max = {runningState.state.velocities.max().cpu().item():.4g} (c0 = {schemeConfig.fluid.fixedSoundSpeed:.4g}), rho_max = {runningState.state.densities.max().cpu().item():.4g}, rho_min = {runningState.state.densities.min().cpu().item():.4g}'
    timingText = f' | iter time: {timing:.3f} ms' if timing is not None else ''

    titleString = f'{caseText} | {timeText} | {particleText} | {domainText} | {obstacleText} | {stateText}{timingText}'
    return titleString


def setupPlotter(runningState, args, simSetup, config, schemeConfig):
    titleString = buildPlotText(runningState, args, simSetup, config, schemeConfig, None)
    markerSize = args.markerSize
    velocityPlot = PlottingOptions(
                colorMap = UniformColorMap.viridis,
                markerSize = markerSize,
                midPoint = 0.0,
                quantityScaling = PlotScaling.Linear,
                mapping = Mapping.L2Norm,
                plotTitle = "Particle Velocity Magnitude",
                plotTitleGap = 0.08,
                boundaryVisualization = VisualizeOptions.Visualize,
                # gridVisualization = GridVisualization(
                #     resolution = 512,
                # ),
                # vMin=1e-10,
                # vMin = 0.0,
                # vMax = schemeConfig.fluid.fixedSoundSpeed * 0.1,
            )
    densityPlot = PlottingOptions(
                colorMap = DivergingColorMap.RdBu,
                flipColorMap=True,
                markerSize = markerSize,
                midPoint = 1.0,
                quantityScaling = PlotScaling.Linear,
                plotTitle = "Particle Density",
                # vMin = 0.95,
                # vMax = 1.05,
                plotTitleGap = 0.08,
                # gridVisualization = GridVisualization(
                #     resolution = 512,
                # ),
            )
    UIDPlot = PlottingOptions(
                colorMap = CyclicColorMap.twilight,
                # flipColorMap=True,
                markerSize = markerSize,
                # midPoint = 1.0,
                quantityScaling = PlotScaling.Linear,
                plotTitle = "Particle IDs",
                # vMin = 0.95,
                # vMax = 1.05
                plotTitleGap = 0.08,
                # gridVisualization = GridVisualization(
                #     resolution = 512,
                # ),
            )

    plotter = visualize(
        particleState = runningState.state,
        domain = config.domain,
        quantities = {
            "A": runningState.state.velocities,
            "B":runningState.state.densities,
            "C":runningState.state.UIDs
        } if args.plotDensity else {
            "A": runningState.state.velocities,
            "B": runningState.state.UIDs
        },
        plotOptions = {
            "A": velocityPlot,
            "B": densityPlot,
            "C": UIDPlot
        } if args.plotDensity else {
            "A": velocityPlot,
            "B": UIDPlot
        },
        figTitle = titleString,
        mosaic = 'ABC' if args.plotDensity else 'AB',
        figsize= (args.plotWidth,args.plotHeight),
        backend='vispy',
        # backend='pyVista',
        # backendOptions = {
        #     # In notebooks, use trame for reliable live updates.
        #     'jupyter_backend': 'trame',
        # }
    )

    plotter.updateTitle(titleString)
    return plotter

def updatePlot(plotter, runningState, args, simSetup, config, schemeConfig, timing):
    titleString = buildPlotText(runningState, args, simSetup, config, schemeConfig, timing)
    
    plotter.updateTitle(titleString)
    plotter.updateQuantities(
        {
            "A": runningState.state.velocities,
            "B": runningState.state.densities,
            "C": runningState.state.UIDs
        } if args.plotDensity else {
            "A": runningState.state.velocities,
            "B": runningState.state.UIDs
        },
        newParticleState = runningState.state,
    )