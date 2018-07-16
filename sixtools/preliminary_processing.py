import numpy as np

# This dictionary should eventually disappear, we should have this region
# dictionary saved as a configuration attribute of rixscam. At this point
# we should change the default for regions in extract_regions to None.
# Also the manual is really confusing regarding what numbers need to go here,
# we should discuss this.
regions = {'dark_low2theta': np.s_[2:1635, 4:1601],
           'data_low2theta': np.s_[1651:3286, 4:1601],
           'dark_high2theta': np.s_[2:1635, 4:1601],
           'data_high2theta': np.s_[1651:3286, 4:1601]}


def extract_region(raw_data, region, regions=regions):
    '''Extracts the data and dark regions from a set of raw detector arrays.

    This function takes in a generator which yields a raw rixscam image and
    returns a generator that yields a list containing the 2D numpy
    array for each region defined by 'region'.

    Parameters
    ----------
    raw_data : generator
        This is a generator that yields a list of 2D arrays (as given by
        db[XXXX]data).
    region : string
        The name of the region defined in 'regions' that we need to extract.
    regions : dict
        A dictionary containing the region names as keys of the form:
        .. code-block::
            {region_1: [slice(x_min_1, x_max_1), slice(y_min_1, y_max_1)], ...
             region_n: [slice(x_min_n, x_max_n), slice(y_min_n, y_max_n)}

    Returns
    -------
    out_generator : generator
        A generator that yields a list of arrays for each list in 'raw_data'
        containing the data from 'region'.

    Examples
    --------
    This is intended to be used in the following, one line, call with XXXX
    being a scan_id or equivalent (start.rixscam.regions is suppose to point
    to the metadata containing the regions for this scan):
    >>> extract_regions(db[XXXX].data, region, db[XXXX].start.rixscam.regions)
    '''
    # step through each of the 'events' ('yields') in raw_data
    for event in raw_data:
        # extract out the first list of arrays
        for arrays in event:
            out_list = []
            for arr in arrays:
                # extract out the region from the array and append to out_list
                out_list.append(arr[regions[region]])

        yield out_list
