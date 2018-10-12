from databroker.assets.handlers_base import HandlerBase
import os.path


class HDF5SingleHandler(HandlerBase):
    '''Handler for hdf5 data stored 1 image per file.
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
    specs = {'AD_HDF5_SINGLE'} | HandlerBase.specs

    def __init__(self, fpath, template, filename, key, frame_per_point=1):
        self._path = os.path.join(fpath, '')
        self._fpp = frame_per_point
        self._template = template
        self._filename = filename
        self._key = key

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
            data = f[self._key]
            ret.append(data)
        return ret

    def get_file_list(self, datum_kwargs):
        ret = []
        for d_kw in datum_kwargs:
            ret.extend(self._fnames_for_point(**d_kw))
        return ret


class AreaDetectorHDF5SingleHandler(HDF5SingleHandler):
    '''Handler for hdf5 data stored 1 image per file by areadetector
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
        super(AreaDetectorHDF5SingleHandler, self).__init__(
            fpath=fpath, template=template, filename=filename,
            key=hardcoded_key, frame_per_point=frame_per_point)


db.reg.register_handler('AD_HDF5_SINGLE', AreaDetectorHDF5SingleHandler)
