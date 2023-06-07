# Paper Repository
 This repository contains the code to reproduce the numerical results in the paper [Posterior Sampling from the Spiked Models via Diffusion Processes](https://arxiv.org/abs/2304.11449). You can cite this work by
 
 ```
@article{montanari2023posterior,
  title={Posterior sampling from the spiked models via diffusion processes},
  author={Montanari, Andrea and Wu, Yuchen},
  journal={arXiv preprint arXiv:2304.11449},
  year={2023}
}
```

# Required packages

To run the code, you will need to install Python 3.9, numpy, scipy and matplotlib

# Usage

Below we illustrate how to reproduce the figures in our paper. Users may want to specify their own values of the problem dimension ``n``, the signal-to-noise ratio ``beta``, the number of steps ``L``, and the step size ``delta``. 

## Trajectory plot (Figure 1)

After loading the functions and classes defined in ``main.py``, run 

```
sim = Simulation(n=1000, delta=0.02, L=500)
sim.plot_trajectory()
```

## Histogram (Figure 2)

Run

```
sim = Simulation(beta=b, n=1000, delta=0.02, L=500)
sim.get_distribution()
distribution_plot(b)
```

for all ``b`` in ``{1.0, 1.1, 1.2, 1.3, 1.4, 1.5}``. 

## Score function plot (Figure 3)

After running 

```
sim = Simulation(beta=b, n=1000, delta=0.01, L=l, N1=300, N2=1, score=True)
sim.run_sim()
```
for all ``b`` in ``{1.1, 1.2, 1.3, 1.4, 1.5}`` and all ``l`` in ``\{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100\}``, call the following function

```
score_plot()
```

