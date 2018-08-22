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