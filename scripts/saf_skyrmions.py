import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc
import functools
import matplotlib.pyplot as plt

IMAGE_PATH = "../images/notebooks"
RESULTS_PATH = "../results"


def get_mesh(D: float):
    r = D / 2
    p1 = (-r, -r, 0)
    p2 = (r, r, 22e-9)

    region = df.Region(p1=p1, p2=p2)
    subregions = {
        "bottom": df.Region(p1=p1, p2=(r, r, 10e-9)),
        "spacer": df.Region(p1=(-r, -r, 10e-9), p2=(r, r, 12e-9)),
        "top": df.Region(p1=(-r, -r, 12e-9), p2=(r, r, 22e-9)),
    }

    mesh = df.Mesh(region=region, cell=(3e-9, 3e-9, 1e-9), subregions=subregions)

    return mesh


def Ms_init(pos, Ms, D):
    x, y, z = pos
    r = D / 2
    if (x**2 + y**2) ** 0.5 < r:
        if z < 10e-9:
            return -Ms
        else:
            return Ms
    else:
        return 0


def m_init(pos, d):
    x, y, z = pos
    r = d / 2
    if (x**2 + y**2) ** 0.5 < r:
        return (0, 0, -1)
    else:
        return (0, 0, 1)


def main(D: float, Ms: float):
    mesh = get_mesh(D)
    mesh.mpl.subregions(
        figsize=(18, 6), filename=f"{IMAGE_PATH}/mesh_subregions_{D / 1e-9:.0f}.png"
    )

    d = 40e-9
    Ms_fun = functools.partial(Ms_init, Ms=Ms, D=D)
    m_fun = functools.partial(m_init, d=d)

    # CONSTANTS
    A = 1e-11  # J/m
    sigma = -0.3e-3  # J/m^2
    K = 0.1e6  # J/m^3
    dmi = 0.5e-3  # J/m^2
    system = mm.System(name="saf_skyrmion")
    system.energy = (
        mm.Exchange(A=A)
        + mm.RKKY(sigma=sigma, sigma2=0, subregions=["bottom", "top"])
        + mm.UniaxialAnisotropy(K=K, u=(0, 0, 1))
        + mm.Demag()
        + mm.DMI(D=dmi, crystalclass="Cnv_z")
    )

    norm = {"bottom": Ms_fun, "top": Ms_fun, "spacer": 0}
    system.m = df.Field(mesh, nvdim=3, value=m_fun, norm=norm, valid="norm")

    system.m.sel("y").mpl(
        figsize=(20, 6), filename=f"{IMAGE_PATH}/m_init_{D / 1e-9:.0f}.png"
    )
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 22)
    system.m.sel("y").mpl(ax=ax, filename=f"{IMAGE_PATH}/mcut_init_{D / 1e-9:.0f}.png")
    plt.close()

    md = oc.MinDriver()
    md.drive(system, dirname=f"{RESULTS_PATH}")

    fig, ax = plt.subplots(figsize=(12, 8))
    system.m.sel(z=17e-9).z.mpl.scalar(ax=ax, cmap="bwr", colorbar_label="z-component")
    fig.savefig(f"../images/notebooks/skyrmion_{D / 1e-9:.0f}.png")

    plt.close()


if __name__ == "__main__":
    print(f"Starting simulation....")
    Ds = [300e-9, 375e-9, 450e-9, 525e-9, 600e-9]
    Mss = [380e3, 400e3, 420e3, 440e3]
    for D in Ds:
        for Ms in Mss:
            main(D, Ms)
            print(f"Finished running for {D:.2e} and {Ms:.2e}")
