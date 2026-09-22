import os
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG -- check / edit all of this before running
# ============================================================================

CSV_PATH = "../data/csv_data/saf_relax-results-multi-labeled.csv"
OUT_DIR = "../data/phase_diagram_plots"

# Column names in your CSV. If these don't match, the script will print the
# actual columns it found so you can fix these.
COL_D = "D"          # diameter, assumed nm
COL_MS = "Ms"        # saturation magnetization, assumed kA/m
COL_KU = "Ku"        # anisotropy, assumed MJ/m^3
COL_DMI = "DMI"      # DMI strength, assumed mJ/m^2
COL_LABEL = "phase_label"     # phase label column -- set this explicitly, e.g. "phase"
                      # or "cluster". If left as None, the script tries to
                      # auto-detect a plausible column and prints what it found.

# Exchange stiffness A, in pJ/m. NOT present in your filenames -- this must
# match whatever constant value was fixed in your actual mumax3/OOMMF script.
# 15 pJ/m is a common ballpark for CoFeB-like PMA thin films, but you must
# confirm this against your own simulation input file.
A_PJ_PER_M = 10.0

# ============================================================================
# Unit conversion to SI, for computing Delta (domain wall width) and D_c
# (Bogdanov-Hubert critical DMI). Adjust the multipliers if your columns are
# already in SI or in different units than assumed above.
# ============================================================================

MU0 = 4 * np.pi * 1e-7  # vacuum permeability, T*m/A


def compute_derived_quantities(df: pl.DataFrame) -> pl.DataFrame:
    """Add SI-unit columns, K_eff, Delta (domain wall width), D_c (critical
    DMI), and the two dimensionless ratios D/Delta and dmi/D_c."""
    df = df.with_columns(
        (pl.col(COL_D) * 1e-9).alias("D_SI"),          # nm -> m
        (pl.col(COL_MS) * 1e3).alias("Ms_SI"),          # kA/m -> A/m
        (pl.col(COL_KU) * 1e6).alias("Ku_SI"),          # MJ/m^3 -> J/m^3
        (pl.col(COL_DMI) * 1e-3).alias("dmi_SI"),       # mJ/m^2 -> J/m^2
        pl.lit(A_PJ_PER_M * 1e-12).alias("A_SI"),       # pJ/m -> J/m (constant)
    )

    df = df.with_columns(
        (pl.col("Ku_SI") - 0.5 * MU0 * pl.col("Ms_SI") ** 2).alias("K_eff")
    )

    n_bad = df.filter(pl.col("K_eff") <= 0).height
    if n_bad > 0:
        print(f"WARNING: {n_bad} rows have K_eff <= 0 (Ms likely too large "
              f"relative to Ku for this formula to apply -- shape anisotropy "
              f"may dominate). These rows will show up as NaN in Delta/D_c.")

    df = df.with_columns(
        (pl.col("A_SI") / pl.col("K_eff")).sqrt().alias("Delta"),               # domain wall width, m
        ((4 / np.pi) * (pl.col("A_SI") * pl.col("K_eff")).sqrt()).alias("D_c"),  # critical DMI, J/m^2
    )

    df = df.with_columns(
        (pl.col("D_SI") / pl.col("Delta")).alias("confinement_ratio"),  # D / Delta
        (pl.col("dmi_SI") / pl.col("D_c")).alias("dmi_ratio"),           # dmi / D_c
    )

    return df


# ============================================================================
# Plot 1: raw phase diagram, D vs Ms, faceted by Ku (rows) and dmi (columns)
# ============================================================================

def plot_raw_phase_diagrams(df: pl.DataFrame, label_col: str, out_path: str):
    ku_values = df[COL_KU].unique().sort().to_list()
    dmi_values = df[COL_DMI].unique().sort().to_list()
    labels = df[label_col].unique().sort().to_list()

    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

    fig, axes = plt.subplots(len(ku_values), len(dmi_values),
                              figsize=(4 * len(dmi_values), 3 * len(ku_values)),
                              squeeze=False)

    for row, ku in enumerate(ku_values):
        for col, dmi in enumerate(dmi_values):
            ax = axes[row][col]
            subset = df.filter((pl.col(COL_KU) == ku) & (pl.col(COL_DMI) == dmi))
            for lab in labels:
                pts = subset.filter(pl.col(label_col) == lab)
                if pts.height == 0:
                    continue
                ax.scatter(pts[COL_D].to_numpy(), pts[COL_MS].to_numpy(),
                           color=color_map[lab], label=str(lab), s=40,
                           edgecolor="k", linewidth=0.3)
            ax.set_title(f"Ku={ku}, dmi={dmi}", fontsize=9)
            if row == len(ku_values) - 1:
                ax.set_xlabel(COL_D)
            if col == 0:
                ax.set_ylabel(COL_MS)

    # single shared legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map[lab], markeredgecolor="k",
                           markersize=8, label=str(lab)) for lab in labels]
    fig.legend(handles=handles, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), title="Phase label")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote raw phase diagram grid -> {out_path}")


# ============================================================================
# Plot 2: dimensionless collapse test. All Ku/dmi values pooled into ONE
# plot, axes = D/Delta and dmi/D_c. If this is the right scaling, points
# should separate by phase cleanly here without needing separate panels.
# ============================================================================

def plot_collapse_test(df: pl.DataFrame, label_col: str, out_path: str):
    labels = df[label_col].unique().sort().to_list()
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

    plt.figure(figsize=(7, 6))
    for lab in labels:
        pts = df.filter(pl.col(label_col) == lab)
        plt.scatter(pts["confinement_ratio"].to_numpy(), pts["dmi_ratio"].to_numpy(),
                    color=color_map[lab], label=str(lab), s=45,
                    edgecolor="k", linewidth=0.3, alpha=0.85)

    plt.xlabel(r"Confinement ratio $D/\Delta$")
    plt.ylabel(r"DMI ratio $dmi/D_c$")
    plt.title("Dimensionless collapse test (all Ku, dmi pooled)")
    plt.legend(title="Phase label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote collapse-test plot -> {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Check CONFIG.")

    df = pl.read_csv(CSV_PATH)
    print(f"Loaded {df.height} rows.")
    print(f"Columns found: {df.columns}")

    # --- resolve label column ---
    label_col = COL_LABEL
    if label_col is None:
        candidates = [c for c in df.columns if c.lower() in
                      ("phase", "label", "cluster", "phase_label", "class")]
        if not candidates:
            raise ValueError(
                "Could not auto-detect a phase-label column. Set COL_LABEL "
                f"explicitly in the CONFIG block. Available columns: {df.columns}"
            )
        label_col = candidates[0]
        print(f"Auto-detected label column: '{label_col}' "
              f"(set COL_LABEL explicitly if this is wrong)")

    # --- sanity checks on the other required columns ---
    for col in (COL_D, COL_MS, COL_KU, COL_DMI):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found. "
                              f"Available columns: {df.columns}")

    print(f"Ku values found: {df[COL_KU].unique().sort().to_list()}")
    print(f"dmi values found: {df[COL_DMI].unique().sort().to_list()}")
    print(f"Phase labels found: {df[label_col].unique().sort().to_list()}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Plot 1: raw phase diagrams ---
    plot_raw_phase_diagrams(df, label_col, os.path.join(OUT_DIR, "phase_diagrams_raw.png"))

    # --- Plot 2: dimensionless collapse test ---
    df = compute_derived_quantities(df)
    plot_collapse_test(df, label_col, os.path.join(OUT_DIR, "collapse_test.png"))

    # save the derived-quantity table too, useful for the regression/SISSO step later
    out_csv = os.path.join(OUT_DIR, "data_with_derived_quantities.csv")
    df.write_csv(out_csv)
    print(f"Wrote data + derived quantities -> {out_csv}")


if __name__ == "__main__":
    main()
