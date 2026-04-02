# Binary Planets

Simulation and analysis code accompanying Cukier & Daylan (in prep.). Injects single and binary planets into compact multi-planet systems and measures dynamical survival rates.

## Requirements

```bash
pip install numpy matplotlib scipy tqdm cmasher rebound
```

The plotting scripts also require a LaTeX installation for rendered labels (`text.usetex = True`).

## Getting the data

There are two ways to obtain the simulation data needed for the plots.

### Option 1 — Download the data from Cukier & Daylan (in prep.)

Run the download script from the repository root:

```bash
python download_data.py
```

This downloads and extracts the collated simulation results into:

```
output/bin_inj/<planet_name>/collated_results.npz
output/sin_inj/<planet_name>/collated_results.npz
```

The zip files are deleted automatically after extraction.

### Option 2 — Run the simulations yourself

Run `run_planets.py` once per planetary system per mode. The `output/` directory and run-name subdirectory are created automatically.

```bash
python run_planets.py <run_name> <mode> <planet_index> [dt] [t_end]
```

| Argument | Description |
|---|---|
| `run_name` | Output subdirectory under `output/` (system name is appended automatically) |
| `mode` | `0` = no injection, `1` = single planet injection, `2` = binary planet injection |
| `planet_index` | 1-based index into `data/compact_systems_run_composite.npy` |
| `dt` | (optional) integrator timestep in years, default `1e-5` |
| `t_end` | (optional) simulation end time in years, default `1,000,000` |

Each invocation launches 1000 simulations in parallel using `multiprocessing.Pool`. Results are written to `output/<run_name>/<planet_name>/<i>/`.

Once all systems have been simulated, collate the per-run outputs into `collated_results.npz` files:

```bash
python collate.py <run_name>
```

`collate.py` skips any planet whose `collated_results.npz` is already newer than all its run directories.

#### Priors

Priors are defined in two places.

**Existing planet properties — [`binary_planets/utils.py`](binary_planets/utils.py)**

These draw orbital parameters for the known planets in each system from their catalog uncertainties.

| Parameter | Prior |
|---|---|
| Stellar mass | Uniform over catalog uncertainty `[mid+lo, mid+hi]`; if missing, drawn from empirical distribution in `data/st_mass_dist.npy` |
| Planet semi-major axis | Uniform over catalog uncertainty `[mid+lo, mid+hi]` |
| Planet eccentricity | Uniform over catalog uncertainty `[mid+lo, mid+hi]`; fixed at **0** if unknown |
| Planet inclination | Fixed at **π/2** (edge-on) |
| Planet mass | Uniform over catalog uncertainty if known; otherwise forecasted from radius using the Forecaster probabilistic mass–radius relation (Chen & Kipping 2017, ApJ 834 17; [arXiv:1603.08614](https://arxiv.org/abs/1603.08614)) |

To change these, edit the corresponding `get_*` functions in [`binary_planets/utils.py`](binary_planets/utils.py).

**Injected planet/binary properties — [`run_planets.py`](run_planets.py)**

These are drawn fresh for each simulation run. All angles (ω, Ω, inclination, phase) are always drawn from `Uniform(-π, π)`.

The **orbital gap** for each system is the largest dynamical gap among adjacent planet pairs, defined as the pair with the greatest ratio of outer to inner semi-major axis. `gap[0]` and `gap[1]` are the semi-major axes of the inner and outer bounding planets respectively. Injected objects are placed log-uniformly within this range.

**Mode 1 — single planet injection** (lines 80–89 of [`run_planets.py`](run_planets.py)):

| Parameter | Prior |
|---|---|
| Mass | `Uniform(0.38, 72)` Earth masses |
| Semi-major axis | `LogUniform(gap[0], gap[1])` AU |
| Eccentricity | `Uniform(0, 1)` |

**Mode 2 — binary planet injection** (lines 91–113 of [`run_planets.py`](run_planets.py)):

| Parameter | Prior |
|---|---|
| Total mass | `Uniform(0.38, 72)` Earth masses |
| Mass ratio q (= m₁/m_tot) | `Uniform(0, 0.5)` |
| COM semi-major axis | `LogUniform(gap[0], gap[1])` AU |
| Binary separation d | `Uniform(0, r_Hill)` (within the Hill radius) |
| Binary eccentricity | `Uniform(0, 1)` |
| COM eccentricity | `Uniform(0, 1)` |

To change a prior, edit the corresponding `np.random.*` call in [`run_planets.py`](run_planets.py) at the line numbers above.

## Making the plots

Plotting scripts can be run from either the repository root or the `plotting/` directory. Figures are saved to `plotting/figs/`.

```bash
python plotting/survival_rates.py      # figures 3, 4, 7
python plotting/single_inj_corner.py   # figure 5
python plotting/binary_inj_corner.py   # figure 6
```

Each script automatically uses the intersection of planets present in both `output/sin_inj/` and `output/bin_inj/`.
