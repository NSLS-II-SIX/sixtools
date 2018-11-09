from databroker.assets.handlers_base import HandlerBase
import os.path
import pandas as pd


class AreaDetector_HDF5SingleHandler_DataFrame(HandlerBase):
    '''Handler for hdf5 data stored 1 image per file and returned as a
    Pandas.DataFrame.

    This will work with all hdf5 files that are a mxn arrays and the data is
    'table like' where m is the number of columns and n is the number of rows.

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
    column_names : list[str]
        The column names of the table
    frame_per_point : float
        the number of frames per point.
    '''
    specs = {'AD_HDF5_SINGLE_DATAFRAME'} | HandlerBase.specs

    def __init__(self, fpath, template, filename, key='/entry/data/data',
                 column_names=None, frame_per_point=1):
        # I have included defaults for `key` and 'column_names' for back
        # compatibility with existing files at SIX.
        self._path = os.path.join(fpath, '')
        self._fpp = frame_per_point
        self._template = template
        self._filename = filename
        self._key = key
        self._column_names = column_names

    def _fnames_for_point(self, point_number):
        start = int(point_number * self._fpp)
        stop = int((point_number + 1) * self._fpp)
        for j in range(start, stop):
            yield self._template % (self._path, self._filename, j)

    def __call__(self, point_number):
        ret = []
        import h5py
        for fn in self._fnames_for_point(point_number):
            with h5py.File(fn, 'r') as f:
                dataframe = pd.DataFrame(f[self._key][:],
                                         columns=self._column_names)
            ret.append(dataframe)
        return ret

    def get_file_list(self, datum_kwargs):
        ret = []
        for d_kw in datum_kwargs:
            ret.extend(self._fnames_for_point(**d_kw))
        return ret
