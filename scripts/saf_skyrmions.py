import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc
import functools
import matplotlib.pyplot as plt
import json

IMAGE_PATH = "../images/saf_skyrmion-0.8nm-z_0.8nm-d_1"
RESULTS_PATH = "../results/saf_skyrmion-0.8nm-d_1"


def get_mesh(D: float, w: float):
    r = D / 2
    p10 = (-r, -r, 0)
    p20 = (r, r, 16.8e-9)

    region = df.Region(p1=p10, p2=p20)
    subregions = {
        "bottom": df.Region(p1=p10, p2=(r, r, 8e-9)),
        "spacer": df.Region(p1=(-r, -r, 8e-9), p2=(r, r, (8 + w) * 1e-9)),
        "top": df.Region(p1=(-r, -r, (8 + w) * 1e-9), p2=p20),
    }

    mesh = df.Mesh(region=region, cell=(3e-9, 3e-9, 8e-10), subregions=subregions)

    return mesh


def Ms_init(pos, Ms, D):
    x, y, z = pos
    r = D / 2
    if (x**2 + y**2) ** 0.5 < r:
        if z < 8e-9:
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


def main(D: float, Ms: float, w: float):
    mesh = get_mesh(D, w)
    mesh.mpl.subregions(
        figsize=(18, 6),
        filename=f"{IMAGE_PATH}/mesh_subregions_{D / 1e-9:.0f}_{Ms / 1e3:.0f}.png",
    )

    d = 40e-9
    Ms_fun = functools.partial(Ms_init, Ms=Ms, D=D)
    m_fun = functools.partial(m_init, d=d)

    # CONSTANTS
    A = 1e-11  # J/m
    sigma = -0.3e-3  # J/m^2
    K = 0.1e6  # J/m^3
    dmi = 1e-3  # J/m^2
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
        figsize=(20, 6),
        filename=f"{IMAGE_PATH}/m_init-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}k_Am.png",
    )
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 18)
    system.m.sel("y").mpl(
        ax=ax,
        filename=f"{IMAGE_PATH}/mcut_init-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}k_Am.png",
    )
    plt.close()

    md = oc.MinDriver()
    md.drive(
        system,
        dirname=f"{RESULTS_PATH}/saf_skyrmion-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}kA_m",
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    system.m.sel(z=12e-9).z.mpl.scalar(ax=ax, cmap="bwr", colorbar_label="z-component")
    system.m.sel(z=12e-9).resample((25, 25)).mpl.vector(
        ax=ax, use_color=False, color="black"
    )

    fig.savefig(f"{IMAGE_PATH}/skyrmion-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}k_Am.png")

    plt.close('all')

    info_m0_dict = {}
    info_m0_dict["Ms"] = f"{Ms / 1e3:.0f}"
    info_m0_dict["D"] = f"{D / 1e-9:.0f}"
    info_m0_dict["w"] = w
    with open(
        f"{RESULTS_PATH}/saf_skyrmion-{D / 1e-9:.0f}nm-{Ms / 1e3:.0f}kA_m/m0.json", "w"
    ) as fp:
        json.dump(info_m0_dict, fp)


if __name__ == "__main__":
    print(f"Starting simulation....")
    # Ds = [300e-9, 375e-9, 450e-9, 525e-9, 600e-9]
    # Mss = [340e3, 360e3, 380e3, 400e3, 420e3, 440e3]
    # Ds = [150e-9, 225e-9]
    #Ds = range(150, 600, 75)
    Ds = range(675, 825, 75)
    Mss = range(260, 460, 20)
    for D in Ds:
        for Ms in Mss:
            if D==675 and Ms in range(260,300,20):
                continue
            main(D * 1e-9, Ms * 1e3, w=0.8)
            print(f"Finished running for {D:.2e} and {Ms:.2e}")
