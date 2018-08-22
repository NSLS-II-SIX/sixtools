import numpy as np
from scipy.interpolate import interp1d
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
        The order of polynominal used is set by len(curvature) - 1
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
             background=None,
             detector='rixscam_image'):
    """
    Create rixs spectra according to procces_dict
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
        The order of polynominal used is set by len(curvature) - 1
    bins : int or array_like or [int, int] or [array, array]
        The bin specification in y then x order:
            * If bins is None a step of 1 is assumed over the relevant range
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
    detector : string
        name of the detector passed on header.data

    Yields
    -------
    ImageStack : pims ImageStack or list of ImageStack
        Array-like object contains scans associated with an event.
        If the input is a list of headers the output is a list of
    """
    for ImageStack in header.data(detector):
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
        The order of polynominal used is set by len(curvature) - 1
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


def calibrate(scan, elastics=None, energy_per_pixel=1, I0s=None):
    """Apply energy per pixel, I0 and energy zero calibration.

    Parameters
    ---------
    scan : array
        4D array of RIXS spectra with structure
        event, image_index, y, I
    elastics : array
        Elastic pixels to subtract to set energy zero
        2D array with shape (event, images per event)
    energy_per_pixel : float
        Multiply all pixel (y) values by this number
        to convert pixel index to energy loss
    I0s : array
        Intensity motor to divide all intensities by
        2D array with shape (event, images per event)

    Returns
    -------
    scan_out : array
        calibrated scans
        4D array of RIXS spectra with structure
        event, image_index, y, I
    """
    if elastics is None:
        elastics = np.zeros(scan.shape[0:2])
    if I0s is None:
        I0s = np.ones(scan.shape[0:2])

    scan_out = scan - elastics[:, :, np.newaxis, np.newaxis]
    scan_out[:, :, :, 0:1] *= energy_per_pixel
    scan_out[:, :, :, 1:2] /= I0s[:, :, np.newaxis, np.newaxis]
    return scan_out


def interp_robust(x, xp, fp):
    """
    Wrapper around scipy to interpolate data with either
    increasing or decreasing x

    Parameters
    ----------
    x : array
        values to interprate onto
    xp : array
        original x values
    fp : array
        original values of function

    Returns
    -------
    f : array
        values interpolated at x
    """
    func = interp1d(xp, fp, bounds_error=False, fill_value=np.NaN)
    f = func(x)
    return f
