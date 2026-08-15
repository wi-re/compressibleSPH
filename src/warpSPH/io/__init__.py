"""Public I/O namespace: writing a run to disk (`export`), reading it back
(`importIO`), the shared HDF5 primitives both use (`hdf5`), string->enum
parsing for CLI/config values (`parsers`), and a `torch.utils.data.Dataset`
over exported trajectories for offline consumption (`dataset`).
"""

from .dataset import DatasetParams, SPHDataset, sph_collate_variable, sample_to_state, sample_to_domain, restore_config_from_h5
from .importIO import importConfigs, importSimulationSystem, loadTrajectory, loadTrajectoryFrame
from .export import prepExport, exportSimulationSystem, writeInitialData, writeFrame
from .hdf5 import createOutFile, copy_dict_to_h5, restore_config_from_h5, restoreConfig_from_h5
from .export import latestExportPath, findExportRuns, exportDirName, resolveExportRoot

__all__ = [
    'DatasetParams', 'SPHDataset', 'sph_collate_variable', 'sample_to_state', 'sample_to_domain', 'restore_config_from_h5',

    'prepExport', 'importConfigs', 'exportSimulationSystem', 'importSimulationSystem',
    'loadTrajectory', 'loadTrajectoryFrame',
    'createOutFile', 'writeInitialData', 'writeFrame', 'copy_dict_to_h5', 'restore_config_from_h5', 'restoreConfig_from_h5',
    'latestExportPath', 'findExportRuns', 'exportDirName', 'resolveExportRoot',
]