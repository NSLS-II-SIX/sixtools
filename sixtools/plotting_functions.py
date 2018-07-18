import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_frame(ax, frame, light_ROIs=[], cax=None, **kwargs):
    """
    Plot the whole frame coming out of the detector and check the ROIs

    Parameters
    ----------
    ax : matplotlib axis object
        axis to plot frame on
    frame : array
        2D image to plot
    light_ROIs : list of [slice, slice]
        Regions of interest to plot as white rectangles
    cax : None or matplotlib axis object
        axis to plot colorbar on
        If none a colorbar is created
    **kwargs : dictionary
        passed onto matplotlib.pyplot.imshow.

    Returns
    -------
    art : matplotlib artist
        artist from the plot
    cax : matplotlib axis
        colorbar axis object
    cb : matplotlib colorbar
        colorbar
    """

    defaults = {'vmin': np.nanpercentile(frame, 5),
                'vmax': np.nanpercentile(frame, 95)}

    kwargs.update({key: val for key, val in defaults.items()
                   if key not in kwargs})

    art = ax.imshow(frame, **kwargs)
    if cax is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)

    cb = plt.colorbar(art, cax=cax)

    for light_ROI in light_ROIs:
        row_i = light_ROI[0].start
        row_f = light_ROI[0].stop
        col_i = light_ROI[1].start
        col_f = light_ROI[1].stop

        box = Rectangle((col_i, row_i), (col_f-col_i), (row_f-row_i),
                        facecolor='none', edgecolor='w')
        ax.add_patch(box)

    cax.set_xticks([])
    cax.set_yticks([])
    return art, cax, cb


def plot_scan(ax, scan, event_labels=None, xlabel='Energy loss',
              ylabel='I', legend_kw={}, **kwargs):
    """
    Plot a scan with a nice series of different colors.
    Labeled by event label, which is either provided or
    which defaults to 0, 1, ...
    and frame number 0, 1, ...

    Parameters
    ----------
    ax : matplotlib axis object
        axis to plot frame on
    scan : array
        4D array of RIXS spectra with structure
        event, image_index, y, I
        event labels
    event labels : iterable
        series of labels. This should match scan.shape[0].
        This defaults to 0, 1, 2, ...
    xlabel : string
        label for the x-axis (default 'pixels')
    ylabel : string
        label for the y-axis (default 'I')
    legend_kw : dictionary
        passed onto ax.legend()
    **kwargs : dictionary
        passed onto matplotlib.pyplot.plot.
    """
    num_spectra = scan.shape[0] * scan.shape[1]
    colors = iter(plt.cm.inferno(np.linspace(0, 0.8, num_spectra)))

    if event_labels is None:
        event_labels = range(scan.shape[0])

    for event, event_label in zip(scan, event_labels):
        for i, S in enumerate(event):
            ax.plot(S[:, 0], S[:, 1], color=next(colors),
                    label="{}_{}".format(event_label, i),
                    **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(**legend_kw)
