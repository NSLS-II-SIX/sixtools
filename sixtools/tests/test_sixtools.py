import numpy as np
import xarray as xr
from rixs.process2d import gauss


def make_test_dataset():
    """Make a fake xarray dataset for testing purposes.

    Returns
    -------
    ds : xarray dataset
        dataset that with spectra occuring at different pixels
    """
    centers = np.linspace(500, 530, 20)
    bins = np.arange(300-0.25/2, 700, 0.25) 
    y = (bins[:-1] + bins[1:])/2
    I = np.vstack([gauss(y, 10, center, 5, 0) for center in centers])
    I = I[:,np.newaxis,:]
    pixel = np.vstack([y for _ in centers])
    pixel =  pixel[:,np.newaxis,:]

    ds = xr.Dataset({'intensity': (['centers', 'frame', 'y'], I),
                    'error': (['centers', 'frame', 'y'], np.sqrt(I))},
                    coords={'centers': centers,
                            'frame': np.array([0]),
                            'pixel': (['centers', 'frame', 'y'], pixel)},
                    attrs=dict(event_name='centers',
                               centers=centers,
                               bins=bins,
                               y=y))
    return ds
