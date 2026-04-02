import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

from scipy.stats import gaussian_kde

import matplotlib as mpl

import os
import pathlib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

sys.path.append(str(ROOT))

import cmasher as cmr  # noqa: E402

os.makedirs(HERE / "figs", exist_ok=True)

run_name = sys.argv[1] if len(sys.argv) > 1 else "sin_inj"

def gen_arr(config, bin=True, n=12000, secondary=True, extra_keys = []):
    n_secondary = config["n_secondary"]
    
    if secondary == "sin_inj":
        sec = 1
    elif secondary:
        sec = n_secondary
    
    l = bin*len(config["binary"].keys()) + sec*len(config["secondary_0"].keys()) + 2 + len(extra_keys)
    arr = np.zeros((n, l))*np.nan
    keys = []
    keys.append("stable")
    keys.append("m_star")

    if secondary == "sin_inj":
        for key in config["secondary_0"].keys():
            keys.append(f"{0}/{key}")
    elif secondary:
        for i in range(0, n_secondary):
            for key in config["secondary_0"].keys():
                keys.append(f"{i}/{key}")
    if bin:
        for key in config["binary"].keys():
            keys.append(f"bin/{key}")
    
    for key in extra_keys:
        keys.append(key)
    
    return arr, keys




def add_to_array(arr, config, keys, i, stable, bin=True, n=12000, secondary=True, extra_keys=[], extra_vars=[]):
    if not np.any(arr):
        arr, keys = gen_arr(config, bin=bin, n=n, secondary=secondary, extra_keys=extra_keys)
        
    n_secondary = config["n_secondary"]
    arr[i, 0] = stable

    arr[i, 1] = config["m_star"]
    
    inj = n_secondary-1
    
    j = 2
    if secondary == "sin_inj":
        for key in config["secondary_0"].keys():
            arr[i, j] = config[f"secondary_{inj}"][key]
            j += 1
    elif secondary:
        for k in range(0, n_secondary):
            for key in config["secondary_0"].keys():
                arr[i, j] = config[f"secondary_{k}"][key]
                j += 1
    if bin:
        for key in config["binary"].keys():
            arr[i, j] = config["binary"][key]
            j += 1
    for k in range(len(extra_keys)):
        arr[i, j] = extra_vars[k]
        j += 1
    
    return arr, keys
    
run_systems = np.load(ROOT / "data" / "compact_systems_run_composite.npy", allow_pickle=True)
all_names = [sys["name"] for sys in run_systems if sys["name"]]

bin_planets = {d.name for d in (ROOT / "output" / "bin_inj").iterdir() if d.is_dir() and (d / "collated_results.npz").exists()}
sin_planets = {d.name for d in (ROOT / "output" / run_name).iterdir() if d.is_dir() and (d / "collated_results.npz").exists()}
valid_planets = bin_planets & sin_planets
names = [name for name in all_names if name in valid_planets]
# names = ["Kepler-186"]


is_binary = False

dir = run_name

extra_keys = ["a_f", "max_change"]

arr_t = None
stable_t = None
stable_omega = []
unstable_omega = []
for system in tqdm(names):
    cached_data =  np.load(ROOT / "output" / dir / system / "collated_results.npz", allow_pickle=True)
    cached_summaries = cached_data["summaries"]
    cached_cfgs = cached_data["cfgs"]
    cached_elements = cached_data["elements"]
    n = len(cached_summaries)
    
    arr = None
    keys = None
    system_idx = np.where(np.array(all_names) == system)[0][0]


    j = 0
    stable = np.zeros(n)

    for i in range(0, n):
        cfg = cached_cfgs[i]
        elements = cached_elements[i]

        n_inj = cfg["n_secondary"] - 1

        a_i = elements[0, :, 0]
        a_f = elements[-1, :, 0]

        stable[i] = np.all(((a_i-a_f)/a_i) < 1e100)
        


        arr, keys = add_to_array(arr, 
                                    cfg, 
                                    keys,
                                    j, 
                                    stable[i], 
                                    bin=is_binary, 
                                    secondary="sin_inj",
                                    extra_keys=extra_keys,
                                    extra_vars=[elements[-1, n_inj, 0], np.nanmax((a_i-a_f)/a_i)]
                                )

        if stable[i]:
            stable_omega.append(cfg["secondary_0"]["omega"])
        else:
            unstable_omega.append(cfg["secondary_0"]["omega"])
        j += 1

    arr = arr[:j, :]
    stable = stable[:j]
    stable = np.logical_not(np.logical_not(stable))
    keys = np.array(keys)
    arr[:, 1:] += np.random.random((arr.shape[0], arr.shape[1]-1)) * 1e-20

    arr[:, keys=="0/a"] = ((np.log10(arr[:, keys=="0/a"]) - np.log10(run_systems[system_idx]["gap"][0]))
                                / (np.log10(run_systems[system_idx]["gap"][1]) - np.log10(run_systems[system_idx]["gap"][0])))

    arr[:, keys=="a_f"] = ((np.log10(arr[:, keys=="a_f"]) - np.log10(run_systems[system_idx]["gap"][0]))
                            / (np.log10(run_systems[system_idx]["gap"][1]) - np.log10(run_systems[system_idx]["gap"][0])))


    if not np.any(arr_t):
        arr_t = arr
        stable_t = stable
    else:
        arr_t = np.vstack([arr_t, arr])
        stable_t = np.hstack([stable_t, stable])


param_labels = [r"Mass [M$_\oplus$]", "a (Rescaled)", "e", "Inclination [rad]"]

mpl.rc("font", size=10)
# plt.style.use("dark_background")

n_vars = 3

fig, axs = plt.subplots(ncols=n_vars+1, nrows=n_vars+1, figsize=(7.5, 7.5))
gs = axs[0, 0].get_gridspec()
n = 50

levels = np.logspace(-2, 0.5, 8)

c1 = mpl.colormaps["cmr.sepia"](0.95)
c2 = mpl.colormaps["cmr.lavender"](0.95)

c1_dark = mpl.colormaps["cmr.sepia"](0.6)
c2_dark = mpl.colormaps["cmr.lavender"](0.6)

pairs = [(i, j) for i in range(n_vars+1) for j in range(i+1, n_vars+1)]
good_keys = ['0/m', '0/a', '0/e', '0/inc', "a_f", "max_change"]



for i, j in tqdm(pairs):
        ax = axs[j,i]
        key1 = good_keys[i]
        key2 = good_keys[j]
        
        x = arr_t[arr_t[:, 0]>-1, keys==key1]
        y = arr_t[arr_t[:, 0]>-1, keys==key2]

        x_s = arr_t[arr_t[:, 0]==1, keys==key1]
        y_s = arr_t[arr_t[:, 0]==1, keys==key2]
        
        xs = np.linspace(np.min(x), np.max(x), n)
        ys = np.linspace(np.min(y), np.max(y), n)

        xx, yy = np.meshgrid(xs, ys)
        
        # --- replaced this ---
        # kde_x = gaussian_kde(x)
        # kde_y = gaussian_kde(y)
        # kde_x_s = gaussian_kde(x_s)
        # kde_y_s = gaussian_kde(y_s)
        # density = np.matmul(kde_y(ys).reshape((n,1)), kde_x(xs).reshape((1,n)))
        # density_s = np.matmul(kde_y_s(ys).reshape((n,1)), kde_x_s(xs).reshape((1,n)))

        # --- with this ---
        kde = gaussian_kde(np.vstack([x.ravel(), y.ravel()]))
        # kde_s = gaussian_kde(np.vstack([x_s.ravel(), y_s.ravel()]))

        xy = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(xy).reshape(n, n)
        # density_s = kde_s(xy).reshape(n, n)
        # -------------------

        ax.contourf(xx, yy, density, cmap="cmr.sepia", alpha=0.5)
        # ax.contour(xx, yy, density_s, linewidths=2, cmap="cmr.lavender", levels=3)

        ax.scatter(x_s, y_s, s=0.25, c="k", alpha=0.3)
#         ax.set_xticks([])
#         ax.set_yticks([])
        ax.sharex(axs[n_vars, i])
        if (j != n_vars): ax.tick_params('x', labelbottom=False)
        else: ax.set_xlabel(param_labels[i])

        
        ax.sharey(axs[j, 0])
        if (i != 0): ax.tick_params('y', labelleft=False)
        else: ax.set_ylabel(param_labels[j])
        ax.set_facecolor("white")
            
for i in range(n_vars+1):
    for j in range(i+1, n_vars+1):
        ax = axs[i,j]
        ax.axis("off")
        if [i,j] == [0,n_vars]:
            ax.hist(np.ones(5), color=c1, label="Prior")
            ax.hist(np.ones(5), color=c2, label="Surviving")
            ax.set_xlim(5,10)
            ax.legend(frameon=False)
 
        if [i,j] == [1,n_vars]:
            p1 = ax.hexbin(np.ones(5), np.ones(5), np.ones(5), cmap='cmr.sepia')
            p2 = ax.hexbin(np.ones(5), np.ones(5), np.ones(5), cmap='cmr.rainforest')
            cmr.set_cmap_legend_entry(p1, 'Prior')
            cmr.set_cmap_legend_entry(p2, 'Surviving')
            
            ax.scatter([0], [0], s=0.25, c="k", alpha=0.3, label = "Surviving Run")

            ax.set_xlim(5,10)
#             ax.legend(frameon=False)
        
for i in range(n_vars+1):
    ax = axs[i,i]
    key1 = good_keys[i]
    x = arr_t[arr_t[:, 0]>-1, keys==key1]
    x_s = arr_t[arr_t[:, 0]==1, keys==key1]

    n_bins=25
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Count")

    ax.hist(x, bins=n_bins, color=c1)
    ax.hist(x, bins=n_bins, color=c1_dark, histtype="step")

    ax.hist(x_s, bins=n_bins, color=c2)
    ax.hist(x_s, bins=n_bins, color=c2_dark, histtype="step", lw=2)

    ax.set_yscale("log")

    ax.sharey(axs[0,0])
    ax.sharex(axs[n_vars, i])
    if (i != n_vars): ax.tick_params('x', labelbottom=False)
    else: ax.set_xlabel(param_labels[i])

for i in range(n_vars+1):
    axs[n_vars, i].tick_params(axis='x', rotation=45)
    axs[i, i].tick_params(axis='x', rotation=45)



plt.savefig(HERE / "figs" / "figure5_corner_single.png", dpi=500)
