# SAF Skyrmion at T>0

Lets try to get the skyrmion for the SAF Structure at T>0 now. For the OOMMF calculator we can use two differernt solvers: 

- [UHH_ThetaEvolver](http://www.nanoscience.de/group_r/stm-spstm/projects/temperature/download.shtml)
- [Xf_ThermHeunEvolver](https://kelvinxyfong.wordpress.com/research/research-interests/oommf-extensions/xf_thermheunevolve/)

### UHH_ThetaEvolver

Simple stochastic model that acts as a random perturbation to the effective field $H_{eff}$ in the form of a fluctuation field $h_{fluc}$, which uses the Euler method to solve the LLG equation while implementing this random fluctuation based on gaussian pseudo random numbers.

### Xf_ThermHeunEvolver
Improved version that combines the above model in the fluctuation field
$$
\sqrt{\frac{2\alpha k_B T}{(1+\alpha^2) \gamma \mu_0 M_s V \delta t}}
$$

## Simulation

The simulation is done the same way, except we now solve the LLG equation instead of minimizing the energy. First we define the energy terms
of the system

```python
## ENERGY TERMS
A = 1e-11  # J/m
sigma = -0.3e-3  # J/m^2
K = 0.1e6  # J/m^3
dmi = 1e-3  # J/m^2
temp = 100 # K
system = mm.System(name="saf_skyrmion_hysteresis_llg", T=100)
system.energy = (
    mm.Exchange(A=A)
    + mm.RKKY(sigma=sigma, sigma2=0, subregions=["bottom", "top"])
    + mm.UniaxialAnisotropy(K=K, u=(0, 0, 1))
    + mm.Demag()
    + mm.DMI(D=dmi, crystalclass="Cnv_z")
)
```
Now, we define the dynamics terms
```python
## DYNAMICS
alpha = 0.3
gamma0 = 2.211e5 # m/As
system.dynamics = mm.Precession(gamma0=gamma0) + mm.Damping(alpha=alpha)
```
Finally the usual initial magnetization configuration is defined
```python
## MAGNETIZATION
norm = {"bottom": Ms_fun, "top": Ms_fun, "spacer": 0}
system.m = df.Field(mesh, nvdim=3, value=m_fun, norm=norm, valid="norm")
```
Here we specify the evolver for the equation and a constant seed for generating the random numbers to make the results reproducible. Here the timestep $\delta t$ is important. 
```python
## EVOLVER
evolver = oc.Xf_ThermHeunEvolver(
    min_timestep=timestep,
    max_timestep=timestep,
    uniform_seed=10
)
```
Finally, we finish by specifying the simulation time and the number of steps to use. In this case we simulate for $t=2$ ns. 
```python
## DRIVING
sim_time = 2 # ns
md = oc.TimeDriver(evolver=evolver)
md.drive(
    system,
    t=sim_time*1e-9,
    n=200,
    n_threads=8,
    dirname=f"{RESULTS_PATH_HYSTERESIS}/saf_skyrmion-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}kA_m-200steps-{sim_time}ns-{temp}K"
)
```
Two simulations were ran for different timesteps:

- $\delta t=1\times 10^{-13}$ s and $t_{sim}= 14:51$
![](../images/saf_skyrmion-300nm-280kA_m-2ns-100K-1e-13s.mp4)

- $\delta t=1\times 10^{-14}$ s and $t_{sim}= 2:25:11$
![](../images/saf_skyrmion-300nm-280kA_m-2ns-100K-1e-14s.mp4)

