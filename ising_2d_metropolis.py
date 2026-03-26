import numpy as np
import matplotlib.pyplot as plt

# Lattice size and MC steps
L = 10
N = L * L
mc_steps = 5000
equil_steps = 4000

# Temperature range
temperatures = np.linspace(1.5, 3.5, 30)

# Initialize lattice
def init_lattice(L):
    return np.random.choice([-1, 1], size=(L, L))

# Periodic boundary conditions energy
def delta_energy(spins, i, j):
    L = spins.shape[0]
    S = spins[i, j]
    neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + \
                spins[(i-1)%L, j] + spins[i, (j-1)%L]
    return 2 * S * neighbors

# Measurements
M_list, E_list, C_list, X_list = [], [], [], []

for T in temperatures:
    spins = init_lattice(L)

    M_samples = []
    E_samples = []

    for step in range(mc_steps):
        for _ in range(N):
            i, j = np.random.randint(0, L, 2)
            dE = delta_energy(spins, i, j)

            if dE < 0 or np.random.rand() < np.exp(-dE / T):
                spins[i, j] *= -1

        # After equilibration
        if step >= equil_steps:
            M = np.sum(spins)
            E = 0

            for x in range(L):
                for y in range(L):
                    S = spins[x, y]
                    neighbors = spins[(x+1)%L, y] + spins[x, (y+1)%L]
                    E += -S * neighbors

            M_samples.append(M)
            E_samples.append(E)

    M_samples = np.array(M_samples)
    E_samples = np.array(E_samples)

    # Averages
    M_avg = np.mean(np.abs(M_samples)) / N
    E_avg = np.mean(E_samples) / N

    # Specific heat
    C = (np.var(E_samples) / (T**2)) / N

    # Susceptibility
    X = (np.var(M_samples) / T) / N

    M_list.append(M_avg)
    E_list.append(E_avg)
    C_list.append(C)
    X_list.append(X)

# Global font settings
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})

marker_size = 6
line_width = 3

#  Magnetization
plt.figure()
plt.plot(temperatures, M_list, 'o-', markersize=marker_size, linewidth=line_width)
plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.title("Magnetization vs Temperature")
plt.grid()
plt.tight_layout()
plt.show()

#  Energy
plt.figure()
plt.plot(temperatures, E_list, 'o-', markersize=marker_size, linewidth=line_width)
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.title("Energy vs Temperature")
plt.tight_layout()
plt.show()

# Specific Heat
plt.figure()
plt.plot(temperatures, C_list, 'o-', markersize=marker_size, linewidth=line_width)
plt.xlabel("Temperature")
plt.ylabel("Specific Heat")
plt.title("Specific Heat vs Temperature")
plt.tight_layout()
plt.show()

#  Susceptibility
plt.figure()
plt.plot(temperatures, X_list, 'o-', markersize=marker_size, linewidth=line_width)
plt.xlabel("Temperature")
plt.ylabel("Susceptibility")
plt.title("Susceptibility vs Temperature")
plt.tight_layout()
plt.show()
