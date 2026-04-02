import numpy as np
import rebound
from .constants import *
from .log import *
import sys

def orbital_charcteristics(m1, m2, d, e=0, phase=0, Omega=0, inc=0):
    a1 = m2/(m1+m2)*d
    a2 = m1/(m1+m2)*d
    

    
    r1 = get_separation(a1, e, phase)
    r2 = get_separation(a2, e, phase)

    T1 = np.sqrt(4*np.pi**2/(G_AU_YR_MSUN * (m1+m2)) * (a1+a2)**3)
    T2 = T1
    
    
    v1 = get_velocity(r1, a1, e, T1)
    v2 = get_velocity(r2, a2, e, T2)
    

    
    xy1 = np.array([r1*np.cos(phase), r1*np.sin(phase), 0])
    xy2 = np.array([r2*np.cos(phase+np.pi), r2*np.sin(phase+np.pi), 0])
    
    

    rot = np.array([[np.cos(Omega), -np.sin(Omega)*np.cos(inc), np.sin(Omega)*np.sin(inc)],
                    [np.sin(Omega), np.cos(Omega)*np.cos(inc), -np.cos(Omega)*np.sin(inc)],
                    [0,             np.sin(inc),                np.cos(inc)]])
    
    x_vec1 = np.matmul(rot, xy1)
    x_vec2 = np.matmul(rot, xy2)
    
    v_dir1 = np.cross(x_vec1, np.matmul(rot, [0,0,1]))
    v_dir2 = np.cross(x_vec2, np.matmul(rot, [0,0,1]))
    
    
    v_hat1 = v_dir1/np.linalg.norm(v_dir1)
    v_hat2 = v_dir2/np.linalg.norm(v_dir2)
    
    
    v_vec1 = v_hat1 * v1
    v_vec2 = v_hat2 * v2

    return [x_vec1, v_vec1], [x_vec2, v_vec2]
    
    
def get_period(m1, m2, d, e=0, phase=0):
    m1=m1*MASS_E/MASS_SUN
    m2=m2*MASS_E/MASS_SUN
    
    r1 = m2/(m1+m2)*d
    r2 = m1/(m1+m2)*d
    
    a1 = get_semi_major(r1, e, phase)
    a2 = get_semi_major(r2, e, phase)

    return np.sqrt(4*np.pi**2/(G_AU_YR_MSUN * (m1+m2)) * (a1+a2)**3)
    
    
    
def get_semi_major(r, e, theta):
    return r*(1 + e*np.cos(theta))/(1 - e**2)

def get_separation(a, e, theta):
    return (1 - e**2)/(1- e* np.cos(theta)) * a


def get_velocity(r, a , e, T):
    return 2* np.pi * (a**2) * np.sqrt(1-e**2)/(r*T)


def get_hill_radius(a, e, m, m_star):
    return a * (1-e) * ((m*MASS_E)/(3*m_star*MASS_SUN))**(1/3)




def init_sim(m_star, n_log=1000, integrator="whfast",
             dt=1e-4, n_particles=2, run_notes=False):
    sim=rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.integrator = integrator
    sim.dt = dt
    sim.ri_whfast.safe_mode = 0
    sim.ri_whfast.corrector = 11
    sim.add(m=m_star)
    log = init_log(n_log, n_particles)

    return sim, log
     


def init_binary_planet(sim, m_star, m1, m2, d, a=1, e=0,
                       e_sys=0, phase=0, Omega=0, inc=0,
                       bin_inc=0, run_notes=None):

    inc = np.pi - inc
    bin_inc = np.pi - bin_inc

    m1=m1*MASS_E/MASS_SUN
    m2=m2*MASS_E/MASS_SUN

    sim.move_to_hel()
    [x1, v1], [x2, v2] = orbital_charcteristics(m1, m2, d, inc=inc, e=e, 
                                                phase=phase, Omega=Omega)

    # oribtal charateristics of the combined planetary system
    _, [[x0, y0, z0], [vx0, vy0, vz0]] = orbital_charcteristics(m_star, m1+m2, a,
                                                                inc=bin_inc,
                                                                e=e_sys)
    if run_notes:
        run_notes.write(f"e: {e}\t e_sys:{e_sys}")


    sim.add(m=m1, 
            x=x0+x1[0], y=y0+x1[1], z=z0+x1[2], 
            vx=vx0+v1[0],vy=vy0+v1[1], vz=vz0+v1[2])
    
    sim.add(m=m2,
            x=x0+x2[0], y=y0+x2[1], z=z0+x2[2], 
            vx=vx0+v2[0], vy=vy0+v2[1], vz=vz0+v2[2])
    
    return sim


def init_single_planet(sim, m, a, e, inc, omega, Omega):
    # inc = np.pi-inc
    sim.move_to_hel()
    mass = m * MASS_E/MASS_SUN
    sim.add(m=mass, a=a, e=e, inc=inc, omega=omega, Omega=Omega)
    sim.move_to_com()
    return sim

def simulate(sim, log, t_end, mode):
    n_log = get_log_len(log)
        
    for t in np.linspace(0, 10, 100):
        sim.integrate(t, exact_finish_time=0)
        log, halt = log_elements(sim, log, mode)
        if halt:
            print("Unstable system.  Stopping...", file=sys.stderr)
            return sim, log, halt, t
        log[2] = 1
    
        
    for t in np.linspace(0, t_end, n_log)[1:]:
        sim.integrate(t, exact_finish_time=0)
        log, halt = log_elements(sim, log, mode)
        if halt:
            print("Unstable system.  Stopping...", file=sys.stderr)
            return sim, log, halt, t
    return sim, log, halt, t

    
    

