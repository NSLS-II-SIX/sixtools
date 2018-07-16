import pytest
import numpy as np
from sixtools import preliminary_processing as pre_proc


def create_raw_data():
    # create a generator of 2D numpy aray lists.
    for j in range(0,3):
        # generate a list of 2D numpy arrays.
        out_list = []
        for i in range(0, 3):
            out_list.append(np.arange(1, 26).reshape(5, 5))

        yield out_list


@pytest.fixture
def regions():
    # generate a 'regions' dictionary.
    out_dict = {}
    for i in range(0, 3):
        out_dict['region_{}'.format(i)] = np.s_[1:3, 0:2]

    return out_dict


@pytest.fixture
def expected_extracted_regions():
    # generate the expected output dictionary for extract_regions
    out_dict = {}
    for i in range(0, 3):
        out_dict['region_{}'.format(i)] = [np.array([[6,  7], [11, 12]]),
                                           np.array([[6,  7], [11, 12]]),
                                           np.array([[6,  7], [11, 12]])]

    return out_dict


def test_extract_region(regions, expected_extracted_regions):
    # test extract_region.
    raw_data=create_raw_data() #this is done in this way as I need a
                               #generator object.
    for region in regions:
        expecteds = expected_extracted_regions[region]
        found = next(pre_proc.extract_region(raw_data, region, regions))
        for i in range(0, len(expecteds)):
            expected = expecteds[i]
            actual = found[i]
            assert np.allclose(actual, expected)
