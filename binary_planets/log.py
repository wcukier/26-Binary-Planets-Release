import numpy as np
from scipy.stats import describe
from matplotlib import pyplot as plt
from os import sys

def init_log(n_log, n_particles):
    elements = np.zeros((n_log, n_particles-1, 6)) * np.nan
    distances = np.zeros(n_log) * np.nan

    return elements, distances, 0

def get_log_len(log):
    return log[0].shape[0]


def log_elements(sim, log, mode):
    # sim.status()
    particles = sim.particles
    elements, distances, t_step = log
    n_log, n_particles, _ = elements.shape
    n_particles=len(particles)

    halt = ""
    for i in range(n_particles)[1:]:
        p = particles[i]
        if i == 0:
            primary = particles[2]
        # elif i==1:
        #     primary = particles[1]
        else:
            primary = particles[0]
            
        o = p.orbit(primary=primary)
        elements[t_step, i-1] = [o.a, o.e, o.inc, o.Omega, o.omega, sim.t]
        
        if p.m > 1e-15:
            if (o.a < 0) or (o.a > 5):
                halt += "semi-major axis out of bounds\t"
            elif (o.e < 0) or (o.e > 1):
                halt += "eccentricity out of bounds\t"

    d = particles[1] ** particles[2]
    distances[t_step] = d
    if mode == 2:
        if d > elements[t_step, 0][0]/2:
            halt += "binary not bound\t"


    t_step += 1
    # halt = o.a < 0
    return [elements, distances, t_step], halt

def save_log(log, file="output"):
    elements, distances, _ = log
    np.save(file+"/elements.npy", elements)
    np.save(file+"/distances.npy", distances)
    
def calc_moments(log, file=None):
    elements, _, _= log
    n_log, n_particles, n_elements = elements.shape
    summary_stats = np.zeros((n_particles, n_elements-1, 4))
    for i in range(n_particles):
        for j in range(n_elements-1):
            summary_stats[i,j,:] = [i for i in describe(elements[:,i,j])][-4:]
    if file:
        np.save(file, summary_stats)
    return summary_stats

# def plot_corner(log, file):
#     elements, _, _ = log
#     corner.corner(np.hstack((elements[:,0,:5], elements[:,1,:5])), 
#                   range=[.999]*10, 
#                   labels=["a1", "e1", "inc1", "Omega1", "omega1", 
#                           "a2", "e2", "inc2", "Omega2", "omega2"])
#     plt.savefig(file)
#     plt.close()
    
def get_derivatives(log):
    elements, _, _= log
    decile = int(round(elements.shape[0] / 10))
    first_i = np.mean((elements[:decile-2, :, :5] - elements[2:decile, :, :5]) 
                      / np.mean(elements[:decile-2, :, 5] - elements[2:decile, :, 5]), 
                      axis=0)
    first_f = np.mean((elements[-decile:-2, :, :5] - elements[-decile+2:, :, :5]) 
                      / np.mean(elements[-decile:-2, :, 5] - elements[-decile+2:, :, 5]), 
                      axis=0)
    
    second_i = np.mean((elements[:decile-2, :, :5] -2*elements[1:decile-1, :, :5]
                        +elements[2:decile, :, :5]) 
                      / np.mean(elements[:decile-2, :, 5] - elements[2:decile, :, 5])**2/4, 
                      axis=0)
    
    second_f = np.mean((elements[-decile:-2, :, :5] - 2*elements[-decile+1:-1, :, :5]
                       + elements[-decile+2:, :, :5]) 
                    / np.mean(elements[-decile:-2, :, 5] - elements[-decile+2:, :, 5])**2/4, 
                    axis=0)
    
    return first_i, first_f, second_i, second_f