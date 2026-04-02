"""anim.py
Author: Wolf Cukier
Used to create animation of the n-body simulation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio


def animate_separation(name, sim, t_start, t_end, n_frames=100):
    n_particles = sim.N
    x = np.zeros((2, n_frames))
    y = np.zeros((2, n_frames))

    dt = (t_end - t_start)/n_frames

    for i in tqdm(range(n_frames)):
        sim.integrate(t_start + i*dt, exact_finish_time=0)
        p1 = sim.particles[1]
        p2 = sim.particles[2]
        x_com = (p1.x*p1.m + p2.x*p2.m)/(p1.m+p2.m)
        y_com = (p1.y*p1.m + p2.y*p2.m)/(p1.m+p2.m)

        
        x[0, i] = p1.x - x_com
        y[0, i] = p1.y - y_com
        
        x[1, i] = p2.x - x_com
        y[1, i] = p2.y - y_com
        
        plt.plot(x[0, :i+1], y[0, :i+1], color="r")
        plt.plot(x[0, i], y[0, i], marker = "o", color="r")
        
        plt.plot(x[1, :i+1], y[1, :i+1], color="k")
        plt.plot(x[1, i], y[1, i], marker = "o", color="k")
        plt.text(-.0075,-.0075, f"{(i*dt) % 1:.2f}")

        plt.axis("square")
        plt.xlim(-.01, .01)
        plt.ylim(-0.01, .01)
        plt.tight_layout()
        plt.savefig(f"anim/{i}.png")
        plt.close()

    frames = [imageio.imread(f"anim/{i}.png") for i in range(n_frames)]
    imageio.mimsave(f'{name}.gif', frames, fps=10)

def animate_simulation(name, sim, t_start, t_end, n_frames=100):
    n_particles = sim.N
    x = np.zeros((n_particles,n_frames))
    y = np.zeros((n_particles, n_frames))

    dt = (t_end - t_start)/n_frames

    for i in range(n_frames):
        sim.integrate(t_start + i*dt, exact_finish_time=0)
        for n_p in range(n_particles):
            p = sim.particles[n_p]
            x[n_p, i] = p.x
            y[n_p, i] = p.y
            plt.plot(x[n_p, :i+1], y[n_p, :i+1], color="k")
            plt.plot(x[n_p, i], y[n_p, i], marker = "o", color="k")
        plt.axis("square")
        plt.xlim(.95, 1.05)
        plt.ylim(-.05, .05)
        plt.savefig(f"anim/{i}.png")
        plt.close()

    frames = [imageio.imread(f"anim/{i}.png") for i in range(n_frames)]
    imageio.mimsave(f'{name}.gif', frames, fps=10)