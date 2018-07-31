import pandas as pd
from collections import OrderedDict
import datetime
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycle
from databroker import DataBroker as db
import ipywidgets
from traitlets import TraitError

# Set defaults
pd.set_option('max_rows', 500)
matplotlib.rcdefaults()
markers = cycle(['o', 's', '^', 'v', 'p', '<', '>', 'h'])


# Functions
def stopped(header):
    """ Test if header stopped sucessfully."""
    try:
        status = header.stop['exit_status']
    except KeyError:
        status = 'Python crash before exit'

    if status == 'success':
        return True
    else:
        return False


def get_scan_id_dict(headers):
    """Get order dictionary of headers that exited sucessfully."""
    return OrderedDict([(get_scan_desc(header), header)
                        for header in headers
                        if stopped(header)])


def get_keys(header):
    """Return sorted list of what was scanned."""
    try:
        keys = header.table().keys()
        key_list = sorted(list(keys))
        return key_list
    except AttributeError:
        return []


def get_scanned_motor(header):
    """Return string describing which motor was scanned."""
    try:
        return ' '.join(header.start['motors'])
    except (AttributeError, KeyError):
        return ''


def get_scan_desc(header):
    """Return string describing scan"""
    return '{} {} {}'.format(header.start['scan_id'],
                             header.start['plan_name'],
                             get_scanned_motor(header))


# Widgets

today_string = str(datetime.datetime.now().date())
db_search_widget = ipywidgets.Text(description='DB search',
                                   value='since=\'{}\''.format(today_string))

select_scan_id_widget = ipywidgets.Select(description='Select uid')

select_x_widget = ipywidgets.Dropdown(description='x')

select_y_widget = ipywidgets.Dropdown(description='y')

select_mon_widget = ipywidgets.Dropdown(description='mon')

use_mon_widget = ipywidgets.Checkbox(description='Normalize')

plot_button = ipywidgets.Button(description='Plot')

clear_button = ipywidgets.Button(description='Clear')

baseline_button = ipywidgets.Button(description='Display baseline')

baseline_display = ipywidgets.HTML('Baseline')

# bindings


def wrap_refresh(change):
    """Query the databroker with user supplied text."""
    try:
        query = eval("dict({})".format(db_search_widget.value))
        headers = db(**query)
    except NameError:
        headers = []
        db_search_widget.value += " -- is an invalid search"

    scan_id_dict = get_scan_id_dict(headers)
    select_scan_id_widget.options = scan_id_dict


db_search_widget.on_submit(wrap_refresh)


def wrap_select_scan_id(change):
    """Update x/y plotting options based on chosen scan."""
    keys = get_keys(select_scan_id_widget.value)
    select_x_widget.options = keys
    scanned_motor = get_scanned_motor(select_scan_id_widget.value)
    try:
        usekey = next((key for key in keys if scanned_motor in key), keys[0])
        select_x_widget.value = usekey
    except IndexError:
        pass

    select_y_widget.options = keys
    default_y = 'fccd_stats1_total'
    select_mon_widget.options = keys
    try:
        select_y_widget.value = default_y
    except TraitError:
        pass


select_scan_id_widget.observe(wrap_select_scan_id)


def wrap_plotit(change):
    """Plot the chosen scan."""
    header = select_scan_id_widget.value
    table = header.table()
    x = table[select_x_widget.value].values
    if use_mon_widget.value:
        y = (table[select_y_widget.value].values
             / table[select_mon_widget.value].values)
    else:
        y = table[select_y_widget.value].values

    label = header.start['scan_id']

    ax = plt.gca()
    ax.plot(x, y, marker=next(markers), label=label)
    ax.set_xlabel(select_x_widget.value)
    ax.set_ylabel(select_y_widget.value)
    ax.legend()


plot_button.on_click(wrap_plotit)


def wrap_baseline(change):
    """Print out table of baseline values."""
    header = select_scan_id_widget.value
    baseline_table = header.table(stream_name='baseline')
    baseline_table.index = ['Before', "After"]
    scan_id = header.start['scan_id']
    title = "<strong> Scan_id {} baseline </strong>".format(scan_id)
    baseline_display.value = title + baseline_table.transpose().to_html()


baseline_button.on_click(wrap_baseline)


def wrap_clearit(change):
    """Clear the plot axes."""
    plt.gca().cla()


clear_button.on_click(wrap_clearit)
