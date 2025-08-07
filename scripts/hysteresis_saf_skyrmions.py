import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc
import matplotlib.pyplot as plt
import functools

from saf_skyrmions import get_mesh, Ms_init, m_init

IMAGE_PATH = "../images/hysteresis"
RESULTS_PATH_HYSTERESIS = "../results/saf_skyrmion-0.8nm-d_1-hysteresis"

def main(Ms, D, w):
  mesh = get_mesh(D, w)
  
  d = 40e-9
  Ms_fun = functools.partial(Ms_init, Ms=Ms, D=D)
  m_fun = functools.partial(m_init, d=d)

  # CONSTANTS
  A = 1e-11  # J/m
  sigma = -0.3e-3  # J/m^2
  K = 0.1e6  # J/m^3
  dmi = 1e-3  # J/m^2
  system = mm.System(name="saf_skyrmion_hysteresis")
  system.energy = (
    mm.Exchange(A=A)
    + mm.RKKY(sigma=sigma, sigma2=0, subregions=["bottom", "top"])
    + mm.UniaxialAnisotropy(K=K, u=(0, 0, 1))
    + mm.Demag()
    + mm.DMI(D=dmi, crystalclass="Cnv_z")
  )

  norm = {"bottom": Ms_fun, "top": Ms_fun, "spacer": 0}
  system.m = df.Field(mesh, nvdim=3, value=m_fun, norm=norm, valid="norm")

  md = oc.MinDriver()
  md.drive(
      system,
      dirname=f"{RESULTS_PATH_HYSTERESIS}/saf_skyrmion-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}kA_m",
  )

  fig, ax = plt.subplots(figsize=(12, 8))
  system.m.sel(z=12e-9).z.mpl.scalar(ax=ax, cmap="bwr", colorbar_label="z-component")
  system.m.sel(z=12e-9).resample((25, 25)).mpl.vector(
    ax=ax, use_color=False, color="black"
  )

  fig.savefig(f"{IMAGE_PATH}/skyrmion-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}k_Am.png")

  plt.close('all')

  ## HYSTERESIS SIMULATION

  Hmin = (0, 0, -4 / mm.consts.mu0)
  Hmax = (0, 0, 4 / mm.consts.mu0)

  hd = oc.HysteresisDriver()
  hd.drive(
    system,
    dirname=f"{RESULTS_PATH_HYSTERESIS}/saf_skyrmion-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}kA_m-hysteresis",
    Hsteps=[[(0, 0, 0), Hmax, 10], [Hmax, Hmin, 10], [Hmin, Hmax, 20]]
  )

  system.table.mpl(
    x="Bz_hysteresis", y=["mz"], marker="o", linewidth=2, linestyle="dashed",
    filename=f"{IMAGE_PATH}/hysteresis_curve-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}k_Am.png",
  )

if __name__ == '__main__':
  print(f"Starting simulation....")
  
  Ds = range(150, 450, 75)
  Mss = range(260, 460, 20)
  for D in Ds:
    for Ms in Mss:
      main(D=D * 1e-9, Ms=Ms * 1e3, w=0.8)
      print(f"Finished running for {D:.2e} and {Ms:.2e}")