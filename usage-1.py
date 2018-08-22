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