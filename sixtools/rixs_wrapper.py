import numpy as np
from pims import pipeline
from rixs.process2d import apply_curvature, image_to_photon_events

# Eventually we will create this information from the configuration
# attributes in ophyd.
process_dict_low_2theta = {'light_ROI': [slice(175, 1609), slice(1, 1751)],
                           'curvature': np.array([0., 0., 0.]),
                           'bins': None}

process_dict_high_2theta = {'light_ROI': [slice(175, 1609), slice(1753, 3503)],
                            'curvature': np.array([0., 0., 0.]),
                            'bins': None}


process_dicts = {'low_2theta': process_dict_low_2theta,
                 'high_2theta': process_dict_high_2theta}


@pipeline
def image_to_spectrum(image, light_ROI=[slice(None, None, None),
                                        slice(None, None, None)],
                      curvature=np.array([0., 0., 0.]), bins=None,
                      background=None):
    """
    Convert a 2D array of RIXS data into a spectrum

    Parameters
    ----------
    image : array
        2D array of intensity
    light_ROI : [slice, slice]
        Region of image containing the data
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python
           curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
    bins : int or array_like or [int, int] or [array, array]
        The bin specification in y then x order:
            * If int, the number of bins for the two dimensions (nx=ny=bins).
            * If array_like, the bin edges for the two dimensions
              (y_edges=x_edges=bins).
            * If [int, int], the number of bins in each dimension
              (ny, nx = bins).
            * If [array, array], the bin edges in each dimension
              (y_edges, x_edges = bins).
            * A combination [int, array] or [array, int], where int
              is the number of bins and array is the bin edges.
    background : array
        2D array for background subtraction

    Yields
    ------
    spectrum : array
        two column array of pixel, intensity
    """
    try:
        photon_events = image_to_photon_events(image[light_ROI]
                                               - background[light_ROI])
    except TypeError:
        photon_events = image_to_photon_events(image[light_ROI])

    spectrum = apply_curvature(photon_events, curvature, bins)
    return spectrum


def get_rixs(header, light_ROI=[slice(None, None, None),
                                slice(None, None, None)],
             curvature=np.array([0., 0., 0.]), bins=None,
             background=None):
    """
    Create rixs according to procces_dict
    and return data as generator with similar behavior to
    header.data()

    Parameters
    ----------
    header : databroker header object
        A dictionary-like object summarizing metadata for a run.
    light_ROI : [slice, slice]
        Region of image containing the data
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python
           curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
    bins : int or array_like or [int, int] or [array, array]
        The bin specification in y then x order:
            * If int, the number of bins for the two dimensions (nx=ny=bins).
            * If array_like, the bin edges for the two dimensions
              (y_edges=x_edges=bins).
            * If [int, int], the number of bins in each dimension
              (ny, nx = bins).
            * If [array, array], the bin edges in each dimension
              (y_edges, x_edges = bins).
            * A combination [int, array] or [array, int], where int
              is the number of bins and array is the bin edges.
    background : array
        2D array for background subtraction

    Yields
    -------
    ImageStack : pims ImageStack or list of ImageStack
        Array-like object contains scans associated with an event.
        If the input is a list of headers the output is a list of
    """
    for ImageStack in header.data('rixscam_image'):
        yield image_to_spectrum(ImageStack, light_ROI=light_ROI,
                                curvature=curvature, bins=bins,
                                background=background)


def make_scan(headers, light_ROI=[slice(None, None, None),
                                  slice(None, None, None)],
              curvature=np.array([0., 0., 0.]), bins=None,
              background=None):
    """
    Make 4D array of RIXS spectra with structure
    event, image_index, y, I

    Parameters
    ----------
    headers : databroker header object or iterable of same
        iterable that returns databroker objects
    light_ROI : [slice, slice]
        Region of image containing the data
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python
           curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
    bins : int or array_like or [int, int] or [array, array]
        The bin specification in y then x order:
            * If int, the number of bins for the two dimensions (nx=ny=bins).
            * If array_like, the bin edges for the two dimensions
              (y_edges=x_edges=bins).
            * If [int, int], the number of bins in each dimension
              (ny, nx = bins).
            * If [array, array], the bin edges in each dimension
              (y_edges, x_edges = bins).
            * A combination [int, array] or [array, int], where int
              is the number of bins and array is the bin edges.
    background : array
        2D array for background subtraction

    Returns
    -------
    scan : array
        4D array of RIXS spectra with structure
        event, image_index, y, I
    """
    if hasattr(headers, 'data') is True:
        headers = [headers]

    rixs_generators = [get_rixs(h, light_ROI=light_ROI, curvature=curvature,
                                bins=bins, background=background)
                       for h in headers]

    scan = np.concatenate([np.array([s for s in rg])
                           for rg in rixs_generators])
    return scan


# Probably can be removed if calibrate in implemented properly
def cal_S(S, elastic, energy_per_pixel):
    """Apply elastic subtraction and energy per pixel calibration.

    Parameters
    ----------
    S : array
        2D y, intensity spectrum
    elastic : y value to set to zero
    energy_per_pixel : float
        convert pixel to energy loss

    Returns
    -------
    S : array
        2D y, intensity spectrum
        after calibration
    """
    Sout = np.copy(S)
    Sout[:, 0] -= elastic
    Sout[:, 0] *= energy_per_pixel
    order = np.argsort(Sout[:, 0])
    return Sout[order, :]


# There is doubtless a numpy implementation of this
# but I can't work it out.
def calibrate(scan, elastics, energy_per_pixel):
    """Apply energy per pixel and energy zero calibration.

    Parameter
    ---------
    scan : array
        4D array of RIXS spectra with structure
        event, image_index, y, I
    """
    scan_out = np.array([[cal_S(S, el, energy_per_pixel)
                          for S, el in zip(event, el_event)]
                         for event, el_event in zip(scan, elastics)])
    return scan_out
