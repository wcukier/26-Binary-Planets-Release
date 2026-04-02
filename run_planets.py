from binary_planets.run_model import run_model
from binary_planets.sim import get_hill_radius
from binary_planets.utils import *  # noqa

from os import sys
import time
import os
import numpy as np
from multiprocessing import Pool


global cfg_name
global mode
global pl_num
global dt
global t_end


def _init_worker(cfg, m, p, d, t):
    global cfg_name, mode, pl_num, dt, t_end
    cfg_name = cfg
    mode     = m
    pl_num   = p
    dt       = d
    t_end    = t

compact_sys = np.load("data/compact_systems_run_composite.npy", allow_pickle=True)
# compact_sys = np.load("data/TOI-178.npy", allow_pickle=True)
# compact_sys = np.load("data/Kepler-11.npy", allow_pickle=True)
n_sys = len(compact_sys)
n_runs = 1000
seq = np.random.SeedSequence().generate_state(n_runs)


# def one_run(run_num, config, mode, pl_num, debug=0):
def one_run(run_num, cfg):

    
#     with open(cfg_file) as f:
#         config = json.load(f)
#     if cfg_name:
#         config["name"] = cfg_name

#     cfg = dict(config)
    sys_num = pl_num  # np.random.randint(n_sys)
    batch_num = int(int(run_num) / n_sys) + 1

    system = compact_sys[sys_num]
#     cfg["name"] = f"{cfg['name']}/{system['name']}"
    print(
        f"Sys_Num: {sys_num} Run Number: {run_num}. Run Name: {cfg['name']}. Batch #: {batch_num}",
        file=sys.stderr,
    )

    n_secondary = len(system["a"])

    cfg["n_secondary"] = n_secondary
    cfg["t_end"] = t_end

    if mode == 1:
        cfg["n_secondary"] += 1

    # cfg["n_secondary"] = 0 #DEBUG

    name = cfg["name"]
    for i in range(1):
        cfg["name"] = name
        cfg["m_star"] = float(get_stellar_mass(system))
        print(f"m_star: {cfg['m_star']}", file=sys.stderr)
        semi_majors = get_semimajor(system)
        es = get_eccen(system)
        incs = get_inc(system)
        masses = get_mass(system)
        for j in range(n_secondary):
            cfg[f"secondary_{j}"] = {}
            cfg[f"secondary_{j}"]["m"] = masses[j]
            cfg[f"secondary_{j}"]["a"] = semi_majors[j]
            cfg[f"secondary_{j}"]["e"] = es[j]
            cfg[f"secondary_{j}"]["inc"] = incs[j] - np.pi / 2
            cfg[f"secondary_{j}"]["omega"] = np.random.uniform(-np.pi, np.pi)
            cfg[f"secondary_{j}"]["Omega"] = np.random.uniform(-np.pi, np.pi)

        if mode == 1:
            j = n_secondary
            cfg[f"secondary_{j}"] = {}
            # cfg[f"secondary_{j}"]["m"] = np.random.normal(
            #     np.mean(masses), np.std(masses)
            # )
            cfg[f"secondary_{j}"]["m"] = np.random.uniform(0.38, 72)

            cfg[f"secondary_{j}"]["a"] =10 ** np.random.uniform(
                np.log10(system["gap"][0]), np.log10(system["gap"][1])
            )

            cfg[f"secondary_{j}"]["e"] = np.random.uniform(0, 1)
            cfg[f"secondary_{j}"]["inc"] = np.random.uniform(-np.pi, np.pi)
            cfg[f"secondary_{j}"]["omega"] = np.random.uniform(-np.pi, np.pi)
            cfg[f"secondary_{j}"]["Omega"] = np.random.uniform(-np.pi, np.pi)

        if mode == 2:
            mass_total = np.random.uniform(0.38, 72)
            q = np.random.uniform(0, 0.5)
            cfg["binary"]["m1"] = q * mass_total
            cfg["binary"]["m2"] = (1 - q) * mass_total
            cfg["binary"]["e"] = np.random.uniform(0, 1)
            cfg["binary"]["e_sys"] = np.random.uniform(0, 1)
            cfg["binary"]["phase"] = np.random.uniform(-np.pi, np.pi)

            cfg["binary"]["Omega"] = np.random.uniform(-np.pi, np.pi)
            cfg["binary"]["inc"] = np.random.uniform(
                -np.pi, np.pi
            )  
            cfg["binary"]["bin_inc"] = np.random.uniform(-np.pi, np.pi) 


            cfg["binary"]["a"] = 10 ** np.random.uniform(
                np.log10(system["gap"][0]), np.log10(system["gap"][1])
            )
            r_hill = get_hill_radius(
                cfg["binary"]["a"], cfg["binary"]["e_sys"], mass_total, cfg["m_star"]
            )
            cfg["binary"]["d"] = r_hill * np.random.uniform(0, 1.0)

        else:
            cfg["binary"]["m1"] = 1e-20
            cfg["binary"]["m2"] = 1e-20

        cfg["dt"] = dt
        cfg["integrator"] = "BS"
#         i = 1
#         try:
#             os.mkdir(f"output/{cfg['name']}")
#         except:
#             pass
#         while True:
#             try:
#                 os.mkdir(f"output/{cfg['name']}/{i}")
#                 break
#             except:
#                 i += 1
#         cfg["name"] = f"{cfg['name']}/{i}"

        print(f"system configuration: {cfg}", file=sys.stderr)
        return run_model(cfg, mode)
        

def run_dispatcher(run_num):
    seed = seq[run_num]
    print(f"Seed: {seed}", flush=True)
    np.random.seed(seed)

    cfg = {
        "name": cfg_name,
        "binary": {},
        "n_log": 1000,
    }

    sys_num = pl_num  # np.random.randint(n_sys)
    batch_num = int(int(run_num) / n_sys) + 1

    system = compact_sys[sys_num]
    cfg["name"] = f"{cfg['name']}/{system['name']}"
    
    i = 0
    try:
        os.mkdir(f"output/{cfg['name']}")
    except:
        pass
    while True:
        try:
            os.mkdir(f"output/{cfg['name']}/{i}")
            break
        except Exception as e:
            i += 1
            if i > 40000:
                raise(e)
    cfg["name"] = f"{cfg['name']}/{i}"
    

        
    early_stop = 1
    while early_stop:
        early_stop = one_run(run_num, cfg)
        if early_stop:
            print(f"{run_num}: Early stopped.  Retrying...", file=sys.stderr)
    

if __name__ == "__main__":
    try:
        cfg_name = sys.argv[1]
        mode = int(sys.argv[2])  # 0 for no inj, 1 for single inj, 2 for binary inj
        pl_num = int(sys.argv[3]) - 1  # bsub doesn't allow for an index of 0
        print(
            f"Simulating {compact_sys[pl_num]['name']}, mode = {mode}", file=sys.stderr
        )

        dt    = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
        t_end = float(sys.argv[5]) if len(sys.argv) > 5 else 1_000_000

    except Exception as e:
        print("Error parsing arguments")
        print(f"Error was {e}")
        raise

    os.makedirs(f"output/{cfg_name}", exist_ok=True)

    # for i in range(1000):
    #     one_run(i)

    with Pool(initializer=_init_worker,
              initargs=(cfg_name, mode, pl_num, dt, t_end)) as p:
        p.map(run_dispatcher, range(0, n_runs))
