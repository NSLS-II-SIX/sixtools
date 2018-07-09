from rixs import preliminary_processing as pre_proc

# This dictionary should eventually disappear, we should have this region
# dictionary saved as a configuration attribute of rixscam. At this point
# we should change the default for regions in extract_regions to None.
# Also the manual is really confusing regarding what numbers need to go here,
# we should discuss this.
regions = {'dark1': [2, 1635, 4, 1601], 'image1': [1651, 3286, 4, 1601],
           'dark2': [2, 1635, 4, 1601], 'image2': [1651, 3286, 4, 1601]}


def extract_regions(arrays, regions=regions):
    '''Extracts the images and dark regions from a series of images.

    This function takes in a generator which yields a raw rixscam image and
    returns a generator that yields a dictionary containing the 2D numpy
    array for each region in regions.

    Parameters
    ----------
    arrays : generator
        This is a generator that yields a list of 2D arrays.
    regions : dict
        A dictionary containg the region names as keys and a tuple with the
        form [left, right, top, bottom] as the value.

    Returns
    -------
    out_generator : generator
        A generator that yields a dictionary for each 2D array in 'scans'. The
        dictionary has the form {'region1_name': region1, ......}

    Examples
    --------
    This is intended to be used in the following, one line, call with XXXX
    being a scan_id or equivalent (start.rixscam.regions is suppose to point
    to the metadata containg the regions for this scan):
    >>> extract_regions(db[XXXX].data, db[XXXX].start.rixscam.regions)
    '''
    for arr in arrays:
        yield pre_proc.extract_regions(arr, regions)
