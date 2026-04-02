from binary_planets.anim import *
from binary_planets.constants import *
from binary_planets.log import *
from binary_planets.sim import *

import sys
import json
import time
import os
import datetime





def run_model(config, mode, debug=0):
    with open(f"output/{config['name']}/run_notes.out", 'w+') as run_notes:
        run_notes.write(f"{sys.argv}\n")
        run_notes.write(f"Run start time: {datetime.datetime.now()}\n")
        for key in config.keys():
            run_notes.write(f"{key}: {config[key]}\n")
        run_notes.write("*************************************************\n\n")

        binary = config["binary"]
        n_secondary = config["n_secondary"]

        n_particles = n_secondary + 1
        if mode==2: n_particles += 2

        sim, log = init_sim(config["m_star"],
                            config["n_log"],
                            config["integrator"],
                            config["dt"],
                            n_particles)

        if mode==2:
            sim  = init_binary_planet(  sim,
                                        config["m_star"],
                                        binary["m1"],
                                        binary["m2"],
                                        binary["d"],
                                        binary["a"],
                                        binary["e"],
                                        binary["e_sys"],
                                        binary["phase"],
                                        binary["Omega"],
                                        binary["inc"],
                                        binary["bin_inc"],
                                        run_notes=run_notes
                                        )
#             sim.N_active = 3
#         elif mode==1:
#             sim.N_active = 2
#         else:
#             sim.N_active = 1
            
#         sim.testparticle_type = 1
        
        n_secondary = config["n_secondary"]
        for i in range(int(n_secondary)):
            sec = config[f"secondary_{i}"]
            sim = init_single_planet(sim,
                                     sec["m"],
                                     sec["a"],
                                     sec["e"],
                                     sec["inc"],
                                     sec["omega"],
                                     sec["Omega"])

            run_notes.write(f"Added secondary_{i}.\n")
            run_notes.write(f"# of particles: sim")
        run_notes.write("*************************************************\n\n")
#         sim.status()



    with open(f"output/{config['name']}/config.json", "w+") as cf:
        json.dump(config, cf)

    E0 = sim.energy()
    Lx0, Ly0, Lz0, = sim.angular_momentum()

    if debug == 1:
        return sim

    t0 = time.time()
    _, _, halt, t_end = simulate(sim, log, config["t_end"], mode)
    if t_end < 5: return 1
    else:
        print(t_end, file=sys.stderr)
    t1 = time.time()

    save_log(log, f"output/{config['name']}")

    E1 = sim.energy()
    Lx1, Ly1, Lz1, = sim.angular_momentum()

    E_err = (E0 - E1)/E0
    L_err = np.sqrt(((Lx0 - Lx1)**2 + (Ly0 - Ly1)**2 + (Lz0 - Lz1)**2)
                    /(Lx0**2+ Ly0**2 + Lx0**2+1e-20))
    first_i, first_f, second_i, second_f = get_derivatives(log)
    
    moments = calc_moments(log)

    try:
        o = sim.particles[2].orbit(primary=sim.particles[1])
    except Exception as e:
        print(e)
        o = sim.particles[2].orbit(primary=sim.particles[0])
#     sum_data = np.zeros((26,5)) * np.nan
#     sum_data[0, 0] = t1-t0
#     sum_data[1, 0] = E_err
#     sum_data[2, :2] = [E0, E1]
#     sum_data[3, 0] = L_err
#     sum_data[4, :3] = [Lx0, Ly0, Lz0]
#     sum_data[5, :3] = [Lx1, Ly1, Lz1]
#     sum_data[6:8, :] = first_i[:2]
#     sum_data[8:10, :] = first_f[:2]
#     sum_data[10:12, :] = second_i[:2]
#     sum_data[12:14, :] = second_f[:2]
#     sum_data[14:19, :4] = moments[0]
#     sum_data[19:24, :4] = moments[1]
#     sum_data[24, :] = [o.a, o.e, o.inc, o.Omega, o.omega]
#     sum_data[26, 0] = t_end
#     np.save(f"output/{config['name']}/summary.npy", sum_data)
    
    
    with open(f"output/{config['name']}/summary.json", "w+") as cf:
        sum_data = {}
        
        sum_data["t_elapsed"] = t1-t0
        sum_data["E_Err"] = E_err
        sum_data["E_0"] = E0
        sum_data["E_1"] = E1
        sum_data["L_err"] = L_err
        sum_data["L_i"] = [Lx0, Ly0, Lz0]
        sum_data["L_f"] = [Lx1, Ly1, Lz1]
        sum_data["Final_elements"] = [o.a, o.e, o.inc, o.Omega, o.omega]
        sum_data["t_end"] = t_end
        sum_data["halt"] = halt
        
        json.dump(sum_data, cf)
    
    
    

    with open(f"output/{config['name']}/run_notes.out", 'a+') as summary:
        summary.write(f"{halt}\n")
        summary.write(f"t_end: {t_end}")
        summary.write(f"Time elapsed: {t1-t0}\n")

        summary.write(f"Energy Error: {E_err:.2e}\tE_i: {E0:.2e}\
            \tE_f: {E1:.2e}\n")

        summary.write(f"Angular Mom Err: {L_err:.2e}\t\
                      L_i: ({Lx0:.2e}, {Ly0:.2e}, {Lz0:.2e})\t\
                          L_f: ({Lx1:.2e}, {Ly1:.2e}, {Lz1:.2e})\n")
        
        summary.write(f"Simulation Time: {sim.t}")

    #     summary.write("\nStatistical Moments:\n")
    #     summary.write(f"{moments}\n")
    #     summary.write("\n====Derivatives====\n")
    #     summary.write(f"Initial first derivative: {first_i}\n")
    #     summary.write(f"Final first derivative: {first_f}\n")
    #     summary.write(f"Initial second derivative: {second_i}\n")
    #     summary.write(f"Final second derivative: {second_f}\n")


    #     o = sim.particles[2].orbit(primary=sim.particles[1])
    #     summary.write(f"Binary system orbital elements: a:{o.a}, e:{o.e}\n")
    
    del sim, log
    return 0


    # plot_corner(log, f"output/{config['name']}/corner.png")





if __name__ == "__main__":
    try:
        print(f"Config file: {sys.argv[1]}", file=sys.stderr)
        with open(sys.argv[1]) as f:
            config = json.load(f)
        if len(sys.argv) > 2:
            config["name"] = sys.argv[2]
        print(f"Run Name: {config['name']}", file=sys.stderr)

    except Exception as e:
        print("Error in loading config file")
        print(f"Error was {e}")
        raise()

    if len(sys.argv) > 4:
        key = sys.argv[3]
        val = sys.argv[4]
        # print(f"{key}: {val}", file=sys.stderr)
        # print(f"{config[key]}", file=sys.stderr)
        print(type(val), file=sys.stderr)
        config[key] = float(val)

    

    i = 1
    try:
        os.mkdir(f"output/{config['name']}")
    except:
        pass
    while(True):
        try:
            os.mkdir(f"output/{config['name']}/{i}")
            break
        except:
            i += 1
    config["name"] = f"{config['name']}/{i}"
    


    # T = get_period(config["m1"], config["m2"], config["d"],
    #                config["e"], config["phase"])
    # print(T)
    # animate_separation(f"output/{config['name']}/end", sim,
    #                    config['t_end'], config['t_end']+.2)