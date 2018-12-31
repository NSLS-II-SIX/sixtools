import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pims import pipeline
from rixs.process2d import apply_curvature, image_to_photon_events
from rixs.process2d import step_to_bins

# Eventually we will create this information from the configuration
# attributes in ophyd.
process_dict_low_2theta = {'light_ROI': [175, 1609, 1, 1751],
                           'curvature': np.array([0., 0., 0.]),
                           'bins': 1}

process_dict_high_2theta = {'light_ROI': [175, 1609, 1753, 3503],
                            'curvature': np.array([0., 0., 0.]),
                            'bins': 1}


process_dicts = {'low_2theta': process_dict_low_2theta,
                 'high_2theta': process_dict_high_2theta}


def centroids_to_spectrum(table, light_ROI=[0, np.inf, 0, np.inf],
                          curvature=np.array([0., 0., 0.]), bins=1,
                          min_threshold=-np.inf, max_threshold=np.inf,
                          ADU_per_photon=1):
    """
    Convert a table of centroided events into a spectrum

    Parameters
    ----------
    table : pandas table
        centroided RIXS photon events
    light_ROI : [minx, maxx, miny, maxy]
        Define the region of the sensor to use.
        Events are chosen with minx <= x < maxx and miny <= y < maxy
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python

           curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
        The order of polynominal used is set by len(curvature) - 1
    bins : float or array like
        Binning in the y direction.
        If `bins` is a sequence, it defines the bin edges,
        including the rightmost edge.
        If `bins' is a single number this defines the step
        in the bins sequence, which is created using the min/max
        of the input data. Half a bin may be discarded in order
        to avoid errors at the edge. (Default 1.)
    min_threshold : float
        fliter events below this threshold
        defaults to -infinity to include all events
    max_threshold : float
        fliter events above this threshold
        defaults to +infinity to include all events
    ADU_per_photon : float
        Conversion factor between the input intensity values in tabel
        and photons. (Default is 1)

    Yields
    ------
    spectrum : array
        two column array of pixel, intensity
    """
    choose = np.logical_and.reduce((table['x_eta'] >= light_ROI[0],
                                    table['x_eta'] < light_ROI[1],
                                    table['y_eta'] >= light_ROI[2],
                                    table['y_eta'] < light_ROI[3],
                                    table['sum_regions'] >= min_threshold,
                                    table['sum_regions'] < max_threshold))

    photon_events = table[choose][['x_eta', 'y_eta', 'sum_regions']].values
    photon_events[:, 2] = photon_events[:, 2]/ADU_per_photon
    spectrum = apply_curvature(photon_events, curvature, bins)
    return spectrum


@pipeline
def image_to_spectrum(image, light_ROI=[0, np.inf, 0, np.inf],
                      curvature=np.array([0., 0., 0.]), bins=1,
                      ADU_per_photon=1,
                      min_threshold=-np.inf, max_threshold=np.inf,
                      background=None):
    """
    Convert a 2D array of RIXS data into a spectrum

    Parameters
    ----------
    image : array
        2D array of intensity
    light_ROI : [minx, maxx, miny, maxy]
        Define the region of the sensor to use.
        Events are chosen with minx <= x < maxx and miny <= y < maxy
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python

           curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
        The order of polynominal used is set by len(curvature) - 1
    bins : float or array like
        Binning in the y direction.
        If `bins` is a sequence, it defines the bin edges,
        including the rightmost edge.
        If `bins' is a single number this defines the step
        in the bins sequence, which is created using the min/max
        of the input data. Half a bin may be discarded in order
        to avoid errors at the edge. (Default 1.)
    min_threshold : float
        fliter events below this threshold
        defaults to -infinity to include all events
    max_threshold : float
        fliter events above this threshold
        defaults to +infinity to include all events
    ADU_per_photon : float
        Conversion factor between the input intensity values in tabel
        and photons. (Default is 1)
    background : array
        2D array for background subtraction

    Yields
    ------
    spectrum : array
        two column array of pixel, intensity
    """
    num_rows, num_cols = image.shape
    if light_ROI[1] is np.inf:
        light_ROI[1] = num_cols
    if light_ROI[3] is np.inf:
        light_ROI[3] = num_rows
    choose = (slice(light_ROI[2], light_ROI[3]),
              slice(light_ROI[0], light_ROI[1]))

    x_centers = np.arange(light_ROI[0], light_ROI[1])
    y_centers = np.arange(light_ROI[2], light_ROI[3])

    if background is None:
        ph_e = image_to_photon_events(image[choose],
                                      x_centers=x_centers,
                                      y_centers=y_centers,
                                      min_threshold=min_threshold,
                                      max_threshold=max_threshold)
    else:
        ph_e = image_to_photon_events(image[choose] -
                                      background[choose],
                                      x_centers=x_centers,
                                      y_centers=y_centers,
                                      min_threshold=min_threshold,
                                      max_threshold=max_threshold)

    photon_events = ph_e
    photon_events[:, 2] = photon_events[:, 2]/ADU_per_photon
    spectrum = apply_curvature(photon_events, curvature, bins)
    return spectrum


def get_rixs(header, light_ROI=[0, np.inf, 0, np.inf],
             curvature=np.array([0., 0., 0.]), bins=1, ADU_per_photon=None,
             detector='rixscam_centroids',
             min_threshold=-np.inf, max_threshold=np.inf,
             background=None):
    """
    Create rixs spectra according to procces_dict
    and return data as generator with similar behavior to
    header.data()

    Parameters
    ----------
    header : databroker header object
        A dictionary-like object summarizing metadata for a run.
    light_ROI : [minx, maxx, miny, maxy]
        Define the region of the sensor to use.
        Events are chosen with minx <= x < maxx and miny <= y < maxy
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python

           curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
        The order of polynominal used is set by len(curvature) - 1
    bins : float or array like
        Binning in the y direction.
        If `bins` is a sequence, it defines the bin edges,
        including the rightmost edge.
        If `bins' is a single number this defines the step
        in the bins sequence, which is created using the min/max
        of the input data. Half a bin may be discarded in order
        to avoid errors at the edge. (Default 1.)
    ADU_per_photon : float
        Conversion factor between the input intensity values in table
        and photons. (Default is 1)
    detector : string
        name of the detector passed on header.data
        At SIX
        'rixscam_centroids' is the centroided data, which is the default
        'rixscam_image' is the image data
    min_threshold : float
        fliter events below this threshold
        defaults to -infinity to include all events
    max_threshold : float
        fliter events above this threshold
        defaults to +infinity to include all events
    background : array
        2D array for background subtraction
        Only used for image data.

    Yields
    -------
    spectra : generator
        RIXS spectra are returned as a generator
    """
    if ADU_per_photon is None:
        pgm_en = header.table(stream_name='baseline',
                              fields=['pgm_en']).mean().values.mean()
        if np.isnan(pgm_en):
            pgm_en = header.table(fields=['pgm_en']).mean().values.mean()

        ADU_per_photon = pgm_en * 1.12

    if detector == 'rixscam_centroids':
        try:
            iter(bins)
        except TypeError:
            if np.isinf(light_ROI[3]):
                total_table = pd.concat(t for event in header.data(detector)
                                        for t in event)
                light_ROI[3] = total_table['y_eta'].max()
            bins = step_to_bins(light_ROI[2], light_ROI[3], bins)

        for event in header.data(detector):
                yield [centroids_to_spectrum(table, light_ROI=light_ROI,
                                             curvature=curvature, bins=bins,
                                             min_threshold=min_threshold,
                                             max_threshold=max_threshold,
                                             ADU_per_photon=ADU_per_photon)
                       for table in event]
    elif detector == 'rixscam_image':
        for ImageStack in header.data(detector):
            yield image_to_spectrum(ImageStack, light_ROI=light_ROI,
                                    curvature=curvature, bins=bins,
                                    ADU_per_photon=ADU_per_photon,
                                    min_threshold=min_threshold,
                                    max_threshold=max_threshold,
                                    background=background)
    else:
        raise Warning("detector {} not reconized, but we will try to"
                      "return data in any case.".format(detector))
        for ImageStack in header.data(detector):
            yield image_to_spectrum(ImageStack, light_ROI=light_ROI,
                                    curvature=curvature, bins=bins,
                                    min_threshold=min_threshold,
                                    max_threshold=max_threshold,
                                    background=background)


def make_scan(headers, light_ROI=[0, np.inf, 0, np.inf],
              curvature=np.array([0., 0., 0.]), bins=1,
              ADU_per_photon=1, detector='rixscam_centroids',
              min_threshold=-np.inf, max_threshold=np.inf,
              background=None):
    """
    Make 4D array of RIXS spectra with structure
    event, image_index, y, I

    Parameters
    ----------
    headers : databroker header object or iterable of same
        iterable that returns databroker objects
    light_ROI : [minx, maxx, miny, maxy]
        Define the region of the sensor to use.
        Events are chosen with minx <= x < maxx and miny <= y < maxy
    curvature : array
        The polynominal coeffcients describing the image curvature.
        These are in decreasing order e.g.
        .. code-block:: python

           curvature[0]*x**2 + curvature[1]*x**1 + curvature[2]*x**0
        The order of polynominal used is set by len(curvature) - 1
    bins : float or array like
        Binning in the y direction.
        If `bins` is a sequence, it defines the bin edges,
        including the rightmost edge.
        If `bins' is a single number this defines the step
        in the bins sequence, which is created using the min/max
        of the input data. Half a bin may be discarded in order
        to avoid errors at the edge. (Default 1.)
    ADU_per_photon : float
        Conversion factor between the input intensity values in table
        and photons. (Default is 1)
    detector : string
        name of the detector passed on header.data
        At SIX
        'rixscam_centroids' is the centroided data, which is the default
        'rixscam_image' is the image data
    min_threshold : float
        fliter events below this threshold
        defaults to -infinity to include all events
    max_threshold : float
        fliter events above this threshold
        defaults to +infinity to include all events
    background : array
        2D array for background subtraction
        Only used for image data.

    Returns
    -------
    scan : array
        4D array of RIXS spectra with structure
        event, image_index, y, I
    """
    if hasattr(headers, 'data') is True:
        headers = [headers]

    rixs_generators = [get_rixs(h, light_ROI=light_ROI, curvature=curvature,
                                bins=bins, detector=detector,
                                ADU_per_photon=ADU_per_photon,
                                min_threshold=min_threshold,
                                max_threshold=max_threshold,
                                background=background)
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
