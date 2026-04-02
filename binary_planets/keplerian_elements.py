import numpy as np
from binary_planets.constants import *


def vec_norm(vec):
    return np.sqrt(np.sum(vec**2))

## @Citation https://downloads.rene-schwarz.com/download/M002-Cartesian_State_Vectors_to_Keplerian_Orbit_Elements.pdf
def cart_to_kepler(cart_pos, cart_vel, m_body, m_star):
    mu = GRAVITATIONAL_CONSTANT * (m_star + m_body)
    
    angular_mom =  np.cross(cart_pos, cart_vel)
    e_vec = np.cross(cart_vel, angular_mom)/mu - cart_pos/vec_norm(cart_pos)
    n = np.cross([0,0,1], angular_mom)
    
    true_anom = np.arccos(np.dot(e_vec, cart_pos)
                        /(vec_norm(e_vec)*vec_norm(cart_pos))) 
    if np.dot(cart_pos, cart_vel) < 0:
        true_anom = 2*np.pi - true_anom


    inc = np.arccos(angular_mom[2]/vec_norm(angular_mom))
    e = vec_norm(e_vec)
    
    # eccentric_anom = 2 * np.arctan(np.tan(true_anom/2)/np.sqrt((1+e)/(1-e)))
    
    Omega = np.arccos(n[1]/vec_norm(n))
    
    if n[2] < 0:
        Omega = 2*np.pi - Omega
        
    omega = np.arccos(np.dot(n, e_vec)/(vec_norm(n) * vec_norm(e)))
    if e_vec[2] < 0:
        omega = 2*np.pi - omega
    
    # mean_anom = eccentric_anom - e*eccentric_anom
    
    semi_major = 1/(2/vec_norm(cart_pos) - vec_norm(cart_vel)**2/mu)
    
    return semi_major, e, inc, Omega, omega, true_anom

def keplerian_elements_test():
    pos = np.array([au, 0, 0])
    vel = np.array([0, au*2*np.pi/year, 0])
    
    print(cart_to_kepler(pos, vel, MASS_E, MASS_SUN))