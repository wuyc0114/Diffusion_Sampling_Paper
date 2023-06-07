import numpy as np
from importlib import reload
import argparse
import matplotlib.pyplot as plt
import os
from scipy.optimize import fmin
from mycolorpy import colorlist as mcp
import scipy.interpolate as interp

from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

def interpolate_polyline(polyline, num_points):
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i-1]):
            duplicates.append(i)
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))


parser = argparse.ArgumentParser()
parser.add_argument('--beta', type=float, default=1)
args = parser.parse_args()

color_list=mcp.gen_color(cmap="ocean",n=6)

def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return cycler("color",colors)


class Simulation:
    def __init__(self, n=1000, beta=2, L=200, delta=0.01, K_AMP=20, N1=1, N2=1, num_trajectory=5,
                 num_distribution=1000, verbose=False, score=False):
        self.n = n
        self.beta = beta
        self.L = L
        self.delta = delta
        self.K_AMP = K_AMP
        self.verbose = verbose
        self.N1 = N1
        self.N2 = N2
        self.get_maximum_correlation()
        self.num_trajectory = num_trajectory
        self.num_distribution = num_distribution
        self.score = score

    def run_sim(self):
        self.get_matrix()
        self.get_nu()

        for L in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            self.L = L
            for n1 in range(self.N1):
                samples = np.zeros(self.N2)
                for n2 in range(self.N2):
                    if not self.score:
                        samples[n2] = self.get_cor(self.get_sample(), self.X)
                    else:
                        samples[n2] = self.get_score(self.get_sample())

                if not self.score:
                    with open(f'beta={self.beta}_L={self.L}_delta={self.delta}.txt', 'a') as f:
                        f.write(str(samples)[1:-1])
                        f.write('\n')
                    f.close()
                else:
                    with open(f'beta={self.beta}_L={self.L}_delta={self.delta}_score.txt', 'a') as f:
                        f.write(str(samples)[1:-1])
                        f.write('\n')
                    f.close()
        return

    @staticmethod
    def get_arrow_size(x, y):
        n = np.sqrt(x**2 + y**2)
        return x / n, y / n

    def plot_trajectory(self):
        for beta in [0.0, 1.0, 1.5]:
            self.beta = beta
            self.get_matrix()
            self.get_nu()
            fig, ax = plt.subplots(1, 5, figsize=(20, 3.3))
            plt.setp(ax, yticks=np.arange(-1, 1.1, 0.5))
            delta = 0.01
            for i in range(self.num_trajectory):
                self.m1, self.m2 = self.get_sample(return_trajectory=True)
                A = np.zeros((len(self.m1), 2))
                A[:, 0] = self.m1
                A[:, 1] = self.m2
                B = A
                ax[i].set_xlim(-1.1, 1.1)
                ax[i].set_ylim(-1.1, 1.1)
                # ax[i].plot(self.m1, self.m2, '->', markevery=1, markeredgecolor='k', markersize=4)
                ax[i].plot(B[:, 0], B[:, 1], color='black', linestyle=(0, (1,1)), linewidth=1.7, alpha=0.75)
                ax[i].set_xlabel(r'$m_1$', fontsize=18)
                ax[i].set_ylabel(r'$m_2$', fontsize=18)
                ax[i].plot(0, 0, marker="o", markersize=3, markeredgecolor="black",  markerfacecolor="black", alpha=0.75)
                ax[i].plot(B[-1, 0], B[-1, 1], marker="*", markersize=10, markeredgecolor="red", markerfacecolor="red",
                           linewidth=3)
                ax[i].set_aspect('equal', adjustable='box')

                #for j in range(len(self.m1) - 1):
                #    lx, ly = self.get_arrow_size(self.m1[j + 1] - self.m1[j], self.m2[j + 1] - self.m2[j])
                #    ax[i].arrow(self.m1[j], self.m2[j], delta * lx, delta * ly, head_width=0.04, head_length=0.04,
                #                facecolor='r', edgecolor='r')

            fig.suptitle(r'$\beta=$' + f'{self.beta}, ' + r'$L=$' + f'{self.L}, ' + r'$\delta=$' + f'{self.delta}',
                         fontsize=18)

            plt.tight_layout()
            plt.show()
            plt.savefig(f'trajectory_beta={self.beta}_L={self.L}_delta={self.delta}.pdf')


        return

    def get_distribution(self):
        self.data = []
        self.tdata = []
        self.get_matrix()
        self.get_nu()
        for cc in range(self.num_distribution):
            print(cc)
            xalg = self.get_sample()[0:10]
            m = self.run_AMP(np.zeros(self.n))
            p = (m + 1) / 2
            x = np.random.binomial(1, p[0:10]) * 2 - 1
            self.data.append(np.dot(xalg, self.X[0:10]))
            self.tdata.append(np.dot(x, self.X[0:10]))
        np.savetxt(f'data_beta={self.beta}_L={self.L}_delta={self.delta}.txt', np.array(self.data))
        np.savetxt(f'tdata_beta={self.beta}_L={self.L}_delta={self.delta}.txt', np.array(self.tdata))
        return


    def get_matrix(self):
        self.X = np.random.binomial(1, 0.5, size=self.n) * 2 - 1
        W = np.random.normal(0, 1 / np.sqrt(2 * self.n), size=(self.n, self.n))
        self.A = self.beta / self.n * np.tensordot(self.X, self.X, axes=0) + W + W.T
        return

    def get_nu(self):
        eig, eigv = np.linalg.eig(self.A)
        self.nu = eigv[:, np.argmax(eig)] * np.sqrt(self.n * self.beta**2 * np.abs(self.beta**2 - 1))
        if np.mean(self.nu * self.X) < 0:
            self.nu *= -1
        return

    def get_score(self, x):
        return self.beta / 2 * np.sum(x * (self.A @ x)) / self.n

    def run_AMP(self, y):
        z = self.nu
        m_prev = np.zeros(self.n)
        for k in range(self.K_AMP):
            m = np.tanh(z)
            b = self.beta**2 * np.mean(1 - np.tanh(z)**2)
            z = self.beta * self.A @ m + y - b * m_prev
            m_prev = np.copy(m)
        #print(self.get_cor(m, self.X))
        return m


    def get_sample(self, return_trajectory=False):
        # If return_trajectory=False, return the sample get by the proposed algorithm
        # Otherwise, return trajectory of the first and second coordinates
        if not return_trajectory:
            y = np.zeros(self.n)
            for l in range(self.L):
                w = np.random.normal(0, 1, self.n)
                m = self.run_AMP(y)
                y = y + m * self.delta + np.sqrt(self.delta) * w
            m = self.run_AMP(y)
            return np.random.binomial(1, (m + 1) / 2) * 2 - 1
            #return (m >= 0) * 2 - 1
        else:
            m1 = []
            m2 = []
            y = np.zeros(self.n)
            for l in range(self.L):
                w = np.random.normal(0, 1, self.n)
                m = self.run_AMP(y)
                y = y + m * self.delta + np.sqrt(self.delta) * w
                if l == 0:
                    m1.append(0)
                    m2.append(0)
                else:
                    m1.append(m[0])
                    m2.append(m[1])

            m = self.run_AMP(y)
            m1.append(m[0])
            m2.append(m[1])
            return m1, m2




    def SE(self, q):
        z = np.random.normal(0,1,5000)
        return np.mean(np.tanh(self.beta * np.sqrt(q) * z + self.beta**2 * q)**2)

    def get_maximum_correlation(self):
        q = 0.01
        for _ in range(5000):
            q = self.SE(q)
        self.cor_theory = q
        return

    @staticmethod

    def get_cor(a, b):
        return np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2))












def plot(list):
    path0 = os.getcwd()
    fig, ax = plt.subplots(1, len(list), figsize=(4 * len(list), 4))
    #L = [0, 20, 40, 60, 80]
    #L = [0, 10, 20, 30, 40]
    #L = [50, 100, 150, 200, 250]
    delta = [0.01, 0.01, 0.01, 0.01, 0.01]
    #delta = [0.02, 0.02, 0.02, 0.02, 0.02]
    #delta = [0.002, 0.002, 0.002, 0.002, 0.002]
    #L = [10, 20, 30, 40, 50]
    #L = [1000, 500, 200]
    #delta = [0.01, 0.02, 0.05]
    L = [100, 200, 300, 400, 500]
    #delta = [0.01, 0.01, 0.01, 0.01, 0.01]
    for count, id in enumerate(list):
        path = path0 + f'/setting{id}'
        os.chdir(path)
        beta_vec = np.round(np.arange(0, 5.01, 0.1), 2)
        theory = np.zeros(len(beta_vec))
        lower = np.zeros(len(beta_vec))
        upper = np.zeros(len(beta_vec))
        for i, beta in enumerate(beta_vec):
            sim = Simulation(beta=beta)
            theory[i] = sim.cor_theory
            s = np.loadtxt(f"beta={beta}.txt", dtype=float)
            lower[i] = np.quantile(s, 0.025)
            upper[i] = np.quantile(s, 0.975)

        ax[count].plot(beta_vec, theory, label='Theory')
        ax[count].fill_between(beta_vec, lower, upper, alpha=0.2)
        ax[count].set_xlabel(r'$\beta$', fontsize=16)
        ax[count].set_ylabel(
            r'$\frac{\langle \mathbf{X}, \mathbf{x}^{\mathrm{alg}}\rangle}{\Vert\mathbf{X}\Vert_2 \Vert\mathbf{x}^{\mathrm{alg}}\Vert_2}$',
            fontsize=16)
        ax[count].title.set_text(f'L = {L[count]}, delta = {delta[count]}')

    plt.tight_layout()
    plt.show()
    os.chdir(os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir)))
    return

def distribution_plot(beta):
    fig, ax = plt.subplots()
    #data = np.loadtxt(f'n5/data_beta={beta}_L=500_delta=0.02.txt')
    #tdata = np.loadtxt(f'n5/tdata_beta={beta}_L=500_delta=0.02.txt')
    data = np.loadtxt(f'new/data_beta={beta}_L=500_delta=0.02.txt')
    tdata = np.loadtxt(f'new/tdata_beta={beta}_L=500_delta=0.02.txt')
    count_tdata = np.zeros(11)
    count_data = np.zeros(11)
    x = np.arange(-10, 11, 2)
    for i in range(11):
        count_tdata[i] = np.sum(tdata == 2 * i - 10)
        count_data[i] = np.sum(data == 2 * i - 10)

    ax.bar(x, count_data, color = color_list[4], edgecolor = 'black', width=1.5)
    plt.xticks(x)
    ax.set_title(r'$\beta=$'+f'{beta}', fontsize=16)
    ax.set_xlabel(r'$\langle {\theta}_{\leq 10}, {\theta}_{\leq 10}^{alg} \rangle$', size=16)
    ax.set_ylabel('Counts', size=16)
    ax.plot(x, count_tdata, '-o', color='DarkOrange',  label='Theory', linewidth=2, markersize=7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'distribution_beta={beta}.pdf')
    plt.show()
    return

def I(gamma):
    return gamma - np.mean(np.log(np.cosh(gamma + np.sqrt(gamma) * np.random.normal(0, 1, 10000))))

def Phi(beta, gamma):
    return beta ** 2 / 4 + gamma ** 2 / (4 * beta ** 2) - gamma / 2 + I(gamma)

def get_likelihood(beta):
    func = lambda x: Phi(beta, x)
    lambda_star = fmin(func, 0)
    return func(lambda_star) + beta**2 / 4


def score_plot():
    fig, ax = plt.subplots()
    beta_vec = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    score_theory = np.zeros_like(beta_vec)
    L_vec = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    for b, beta in enumerate(beta_vec):
        score_theory[b] = get_likelihood(beta)
        vector = np.zeros_like(L_vec, dtype=float)
        upper = np.zeros_like(L_vec, dtype=float)
        lower = np.zeros_like(L_vec, dtype=float)
        for i, L in enumerate(L_vec):
            data = np.loadtxt(f'scores/beta={beta}_L={L}_delta=0.01_score.txt')
            vector[i] = np.mean(data)
            upper[i] = np.quantile(data, 0.9)
            lower[i] = np.quantile(data, 0.1)
        #with sns.color_palette("GnBu", n_colors=10):
        ax.plot(L_vec / 100, vector, label=r'$\beta$='+f'{beta}', linewidth=2, color=color_list[b])
        #ax.errorbar(L_vec / 100, vector, yerr=[-lower + vector, upper - vector], fmt='o', linewidth=2, markersize=4,
        #            capsize=5, color=color_list[b])
        ax.fill_between(L_vec / 100, lower, upper, color=color_list[b], alpha=0.15)
        plt.axhline(y=score_theory[b], color=color_list[b], linestyle='--', xmin=0.05, xmax=0.95, alpha=0.7,
                    linewidth=1.4)
    plt.axhline(y=score_theory[b], color='k', linestyle='--', xmin=0.00, xmax=0.00, alpha=0.7,
                linewidth=1.4, label='theory')

    plt.legend()
    #plt.grid(True)
    ax.set_xlabel(r'$T$', fontsize=18)
    ax.set_ylabel(r'$\frac{\beta}{2n}\langle {\mathbf{\theta}^{alg}}, {X}  {\mathbf{\theta}^{alg}} \rangle$', fontsize=15)
    plt.savefig(f'score.pdf')


if __name__ == '__main__':
    beta = args.beta

    ###------Generate Figure 1--------------------------------------------

    sim = Simulation(beta=beta, n=1000, delta=0.02, L=500, N1=1, N2=1, score=True)
    sim.plot_trajectory()
    #sim = Simulation(beta=beta, n=1000, delta=0.02, L=500, N1=1, N2=1, num_trajectory=1)
    sim.run_sim()








