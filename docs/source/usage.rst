=====
Usage
=====

RIXS spectra at SIX are represented as two-column arrays of values, where the axes are energy loss and intensity. If we call such as spectrum ``S``, it can be plotted by


.. code-block:: python

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(S[:,0], S[:,1])

.. plot::

       import matplotlib as mpl
       mpl.rcdefaults()
       import matplotlib.pyplot as plt
       fig, ax = plt.subplots()

       x = np.linspace(-2, 5, 100)
       y = (np.exp(-x**2/ 0.01) + np.exp(-(x-.05)**2/ 0.03) +
            10*np.exp(-(x-2)**2/ 0.1) + 7*np.exp(-(x-2.5)**2/ 0.15)
            + 0.2*np.random.rand(x.size))
       ax.set_xlabel('Energy loss')
       ax.set_ylabel('Intensity (photons)')
       ax.plot(x, y*10, '.-')


The use of BlueSky at SIX allows us to take RIXS spectra at a series of different conditions in a single operation or scan. For example one could vary the scattering angle :math:`\theta` to measure at low, medium and high :math:`\theta`. The scan containing all spectra is retrieved via


.. code-block:: python

    from sixtools.rixswrapper import make_scan
    from databroker in DataBroker as db

    scan = make_scan(db[21110])

``scan`` is then a 4D numpy array. Its axes are:


* axis 0 -- event in scan
* axis 1 -- frame in event
* axis 2 -- row in spectrum
* axis 3 -- axis of spectrum

Since each frame is taken under nominally the same conditions (i.e. the same :math:`\theta` in our example), one might want to add the spectra at each event.

.. code-block:: python

    scan_events_summed = scan.sum(axis=1)

These summed spectra could be plotted by against :math:`\theta`, which we can read out of the databroker. Let's zoom on the low energy region, as this is more likely to be dispersive.

.. code-block:: python

    fig, ax  = plt.subplots()

    thetas = db[21110].table(fields=['theta'])

    for S, theta in zip(scan_events_summed, thetas):
        ax.plot(S[:,0], S[:,1], label='$\theta={}$'.format(theta))

.. plot::

    import matplotlib as mpl
    mpl.rcdefaults()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*']

    x = np.linspace(-2, 5, 100)
    for theta, mag_energy, marker in zip([10, 30, 60], [0.05, 0.2, 0.35], markers):
        y = (np.exp(-x**2/ 0.01) + np.exp(-(x-mag_energy)**2/ 0.03) +
        10*np.exp(-(x-2)**2/ 0.1) + 7*np.exp(-(x-2.5)**2/ 0.15)
        + 0.2*np.random.rand(x.size))
        ax.plot(x, y*10, '-', marker=marker, label=r'$\theta={}$'.format(theta))

    ax.axis([-0.5, 1, -1, 30])
    ax.set_xlabel('Energy loss')
    ax.set_ylabel('Intensity (photons)')
    ax.legend()



or alternatively one can plot all individual spectra

.. code-block:: python

    import matplotlib.pyplot as plt
    fig, ax  = plt.subplots()

    for event, theta in zip(scan, theta):
        for frame_ind, S in enumerate(event):
            ax.plot(S[:,0], S[:,1],
            label='$\theta={} frame {}'.format(theta, frame_ind))

.. plot::

    import matplotlib as mpl
    mpl.rcdefaults()
    import matplotlib.pyplot as plt

    colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = iter(['o', 's', '^', 'v', '<', '>'])


    fig, ax = plt.subplots()

    x = np.linspace(-2, 5, 100)
    for theta, mag_energy, color in zip([10, 30, 60], [0.05, 0.2, 0.35],
                                        colors):
        for frame_ind in range(2):
            y = (np.exp(-x**2/ 0.01) + np.exp(-(x-mag_energy)**2/ 0.03) +
            10*np.exp(-(x-2)**2/ 0.1) + 7*np.exp(-(x-2.5)**2/ 0.15)
            + 0.4*np.random.rand(x.size))
            ax.plot(x, y*10, '.-', color=color, alpha=0.5, marker=next(markers),
            label=r'$\theta={}$ frame {}'.format(theta, frame_ind))

    ax.axis([-0.5, 1, -1, 30])
    ax.set_xlabel('Energy loss')
    ax.set_ylabel('Intensity (photons)')
    ax.legend()

Powerful numpy broadcasting methods can be used to manipulate the whole scan at once. See `Numpy documentation
<https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ or the ``calibrate`` function in ``sixtools.rixswrapper``.

