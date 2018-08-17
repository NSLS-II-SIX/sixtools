=====
Usage
=====

RIXS spectra at SIX are represented as two-column arrays of values, where the axes are energy loss and intensity. If we call such as spectrum `S`, it can be plotted by

.. code-block:: python

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(S[:,0], S[:,1])

The use of BlueSky at SIX allows us to take RIXS spectra at a series of different conditions in a single operation or scan.

.. code-block:: python

    from sixtools.rixswrapper import make_scan
    from databroker in DataBroker as db

    scan = make_scan(db[21110])

`scan` is then a 4D numpy array. Its axes are:

* axis 0 -- event in scan
* axis 1 -- frame in event
* axis 2 -- row in spectrum
* axis 3 -- axis of spectrum

The first spectrum can be obtained and plotted via:

.. code-block:: python

    S = scan[0,0]
    fig, ax = plt.subplots()
    ax.plot(S[:,0], S[:,1])

Since each frame is taken under nominally the same conditions, one might want to add the spectra at each event.

.. code-block:: python

    scan_events_summed = scan.sum(axis=1)

These summed spectra could be plotted by

.. code-block:: python

    fig, ax  = plt.subplots()

    for S in scan_events_summed:
        ax.plot(S[:,0], S[:,1])

or alternatively one can plot all individual spectra

.. code-block:: python

    import matplotlib.pyplot as plt
    fig, ax  = plt.subplots()

    for event in scan:
        for S in event:
            ax.plot(S[:,0], S[:,1])

Powerful numpy broadscating methods can be used to manuipuate the whole scan at once. See `Numpy documentation
<https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ or the `calibrate` function in `sixtools.rixswrapper`.
