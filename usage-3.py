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