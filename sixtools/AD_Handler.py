from databroker.assets.handlers_base import HandlerBase
import os.path
import pandas as pd
import numpy as np


class HDF5SingleHandler_centroid7xn(HandlerBase):
    '''Handler for hdf5 data stored 1 image per file by the rixscam xip plugin.

    This will work with all hdf5 files that are a 7xn array where n is the
    number of centroid 'events' and the 7 'columns' have the format:

        column 1: x
        column 2: y
        column 3: x eta correction
        column 4: y eta correction
        column 5: y eta correction & isolinear correction
        column 6: "sum of 3x3 region" (though, this could also mean 2x2 region)
        column 7: XIP mode (which I assume to be either 3x3 or 2x2)

    This handler will ouput the data as a pandas dataframe with the 'fields'
    (in order): `x`, `y`, `x_eta`, `y_eta`, `y_eta_+_iso`, `sum regions`,
    XIP_mode.

    Parameters
    ----------
    fpath : string
        filepath
    template : string
        filename template string.
    filename : string
        filename
    key : string
        the 'path' inside the file to the data set.
    frame_per_point : float
        the number of frames per point.
    '''
    specs = {'AD_HDF5_SINGLE_C7xn'} | HandlerBase.specs

    def __init__(self, fpath, template, filename, key, frame_per_point=1):
        self._path = os.path.join(fpath, '')
        self._fpp = frame_per_point
        self._template = template
        self._filename = filename
        self._key = key
        self._column_names = ['x', 'y', 'x_eta', 'y_eta', 'y_eta+iso',
                              'sum_regions', 'XIP_mode']

    def _fnames_for_point(self, point_number):
        start = int(point_number * self._fpp)
        stop = int((point_number + 1) * self._fpp)
        for j in range(start, stop):
            yield self._template % (self._path, self._filename, j)

    def __call__(self, point_number):
        ret = []
        import h5py
        for fn in self._fnames_for_point(point_number):
            f = h5py.File(fn, 'r')
            dataframe = pd.DataFrame(np.array(f[self._key]),
                                     columns=self._column_names)
            ret.append(dataframe)
        return ret

    def get_file_list(self, datum_kwargs):
        ret = []
        for d_kw in datum_kwargs:
            ret.extend(self._fnames_for_point(**d_kw))
        return ret


class AreaDetectorHDF5SingleHandler_centroid7xn(HDF5SingleHandler_centroid7xn):
    '''Handler for hdf5 data stored 1 image per file by areadetector from the
    rixscam xip plugin.


    This will work with all hdf5 files that are a 7xn array where n is the
    number of centroid 'events' and the 7 'columns' have the format:

        column 1: x
        column 2: y
        column 3: x eta correction
        column 4: y eta correction
        column 5: y eta correction & isolinear correction
        column 6: "sum of 3x3 region" (though, this could also mean 2x2 region)
        column 7: XIP mode (which I assume to be either 3x3 or 2x2)

    This handler will ouput the data as a pandas dataframe with the 'fields'
    (in order): `x`, `y`, `x_eta`, `y_eta`, `y_eta_+_iso`, `sum regions`,
    XIP_mode.

    Parameters
    ----------
    fpath : string
        filepath
    template : string
        filename template string.
    filename : string
        filename
    frame_per_point : float
        the number of frames per point.
    '''
    def __init__(self, fpath, template, filename, frame_per_point=1):
        hardcoded_key = '/entry/data/data'
        super(AreaDetectorHDF5SingleHandler_centroid7xn, self).__init__(
              fpath=fpath, template=template, filename=filename,
              key=hardcoded_key, frame_per_point=frame_per_point)
