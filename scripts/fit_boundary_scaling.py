import os
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================

CSV_PATH = "../data/csv_data/saf_relax-results-multi-labeled.csv"
OUT_DIR = "../data/boundary_fit_plots"

COL_D = "D"
COL_MS = "Ms"
COL_KU = "Ku"
COL_DMI = "DMI"
COL_LABEL = "phase_label"

# The two labels whose D-axis boundary you want to trace. LABEL_LOW should
# be the phase found at SMALLER D, LABEL_HIGH the phase at LARGER D, for a
# fixed (Ms, dmi, Ku) group -- check your raw phase diagram grid first.
LABEL_LOW = 0   # e.g. "Complejo"
LABEL_HIGH = 1  # e.g. "Laberinto"

# Exchange stiffness, pJ/m -- must match your simulation's actual value.
A_PJ_PER_M = 10.0
MU0 = 4 * np.pi * 1e-7


# ============================================================================
# Derived quantities
# ============================================================================

def add_K_eff(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col(COL_MS) * 1e3).alias("Ms_SI"),
        (pl.col(COL_KU) * 1e6).alias("Ku_SI"),
    )
    df = df.with_columns(
        (pl.col("Ku_SI") - 0.5 * MU0 * pl.col("Ms_SI") ** 2).alias("K_eff")
    )
    return df


# ============================================================================
# Boundary extraction: for each (dmi, Ms, Ku) group, sorted by D, find the
# midpoint D between the last LABEL_LOW point and the first LABEL_HIGH point
# that follows it. Groups without a clean single LABEL_LOW -> LABEL_HIGH
# transition are skipped and counted.
# ============================================================================

def extract_boundaries(df: pl.DataFrame) -> pl.DataFrame:
    group_cols = [COL_DMI, COL_MS, COL_KU]
    boundaries = []
    n_skipped = 0

    for keys, group in df.group_by(group_cols):
        group = group.sort(COL_D)
        d_vals = group[COL_D].to_list()
        labels = group[COL_LABEL].to_list()

        # find last index with LABEL_LOW and first index after it with LABEL_HIGH
        low_idxs = [i for i, l in enumerate(labels) if l == LABEL_LOW]
        high_idxs = [i for i, l in enumerate(labels) if l == LABEL_HIGH]

        if not low_idxs or not high_idxs:
            n_skipped += 1
            continue

        last_low = max(low_idxs)
        candidates_high = [i for i in high_idxs if i > last_low]
        if not candidates_high:
            n_skipped += 1
            continue
        first_high_after = min(candidates_high)

        d_boundary = (d_vals[last_low] + d_vals[first_high_after]) / 2.0

        row = dict(zip(group_cols, keys))
        row["D_boundary"] = d_boundary
        boundaries.append(row)

    print(f"Extracted {len(boundaries)} boundary points, skipped {n_skipped} "
          f"groups with no clean {LABEL_LOW}->{LABEL_HIGH} transition.")

    return pl.DataFrame(boundaries)


# ============================================================================
# Fit both hypotheses and compare
# ============================================================================

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_and_compare(boundaries: pl.DataFrame, out_path: str):
    boundaries = add_K_eff(boundaries)
    boundaries = boundaries.with_columns(pl.col("K_eff").abs().alias("abs_K_eff"))

    x = boundaries["abs_K_eff"].to_numpy()
    y = boundaries["D_boundary"].to_numpy()
    log_y = np.log(y)

    if len(x) < 3:
        print("Not enough boundary points to fit (need >= 3). "
              "Check LABEL_LOW/LABEL_HIGH and your data.")
        return

    # Hypothesis 1: EXPONENTIAL -- log(D_boundary) ~ a * abs_K_eff + b
    a_exp, b_exp = np.polyfit(x, log_y, 1)
    pred_exp = a_exp * x + b_exp
    r2_exp = r_squared(log_y, pred_exp)

    # Hypothesis 2: POWER LAW -- log(D_boundary) ~ n * log(abs_K_eff) + c
    log_x = np.log(x)
    n_pow, c_pow = np.polyfit(log_x, log_y, 1)
    pred_pow = n_pow * log_x + c_pow
    r2_pow = r_squared(log_y, pred_pow)

    print(f"\nEXPONENTIAL fit:  log(D_boundary) = {a_exp:.4g} * |K_eff| + {b_exp:.4g}"
          f"   (R^2 = {r2_exp:.4f})")
    print(f"POWER LAW fit:    log(D_boundary) = {n_pow:.4g} * log(|K_eff|) + {c_pow:.4g}"
          f"   (i.e. D_boundary ~ |K_eff|^{n_pow:.4g})   (R^2 = {r2_pow:.4f})")

    if r2_exp > r2_pow:
        print(f"\n-> Exponential fit is better (R^2 {r2_exp:.4f} vs {r2_pow:.4f}): "
              f"consistent with spin-reorientation-transition-style divergence.")
    else:
        print(f"\n-> Power-law fit is better (R^2 {r2_pow:.4f} vs {r2_exp:.4f}): "
              f"consistent with confinement-ratio-style scaling.")

    # --- plot both fits ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(x, log_y, color="steelblue", edgecolor="k", label="data")
    x_sorted = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_sorted, a_exp * x_sorted + b_exp, color="crimson",
                 label=f"exp fit (R²={r2_exp:.3f})")
    axes[0].set_xlabel(r"$|K_{eff}|$")
    axes[0].set_ylabel(r"$\log(D_{boundary})$")
    axes[0].set_title("Exponential hypothesis")
    axes[0].legend()

    axes[1].scatter(log_x, log_y, color="steelblue", edgecolor="k", label="data")
    logx_sorted = np.linspace(log_x.min(), log_x.max(), 100)
    axes[1].plot(logx_sorted, n_pow * logx_sorted + c_pow, color="darkorange",
                 label=f"power-law fit (R²={r2_pow:.3f})")
    axes[1].set_xlabel(r"$\log(|K_{eff}|)$")
    axes[1].set_ylabel(r"$\log(D_{boundary})$")
    axes[1].set_title("Power-law hypothesis")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nWrote comparison plot -> {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Check CONFIG.")

    df = pl.read_csv(CSV_PATH)
    print(f"Loaded {df.height} rows. Columns: {df.columns}")

    for col in (COL_D, COL_MS, COL_KU, COL_DMI, COL_LABEL):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found. "
                              f"Available columns: {df.columns}")

    os.makedirs(OUT_DIR, exist_ok=True)

    boundaries = extract_boundaries(df)
    if boundaries.height == 0:
        print("No boundaries extracted -- check LABEL_LOW/LABEL_HIGH and "
              "that your D-axis ordering assumption (LOW at small D, HIGH "
              "at large D) actually holds in this data.")
        return

    out_csv = os.path.join(OUT_DIR, "extracted_boundaries.csv")
    boundaries.write_csv(out_csv)
    print(f"Wrote boundary points -> {out_csv}")

    fit_and_compare(boundaries, os.path.join(OUT_DIR, "exp_vs_powerlaw_fit.png"))


if __name__ == "__main__":
    main()
