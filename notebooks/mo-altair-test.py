import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import seaborn as sns
    import polars as pl
    import altair as alt
    import base64
    import mimetypes
    from PIL import Image
    from io import BytesIO
    import os

    return BytesIO, Image, alt, base64, mo, os, pl


@app.cell
def _(BytesIO, Image, base64, os):
    TRANSPARENT_PIXEL = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )

    def to_base64_resized(path, max_size=(128, 128)):
        try:
            if not os.path.exists(path):
                return TRANSPARENT_PIXEL

            with Image.open(path) as img:
                img = img.convert("RGBA")  # ensures compatibility
                img.thumbnail(max_size)   # keeps aspect ratio

                buffer = BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/png;base64,{encoded}"

        except Exception:
            return TRANSPARENT_PIXEL

    return (to_base64_resized,)


@app.cell
def _(pl, to_base64_resized):
    csv_path = '../data/csv_data/saf_relax-results.csv'
    df = pl.read_csv(csv_path).with_columns([
        pl.when(abs(pl.col("S2k_bot") - 1) < 0.25)
          .then(1)
          .otherwise(0)
          .alias("Sk"),
        pl.format(
          r"C:\SPIN-UNI\Orlando\micromagnetics\images\saf_results_relax\bottomlayer_D={}_Ms={}_T=0_dmi={}_Ku={}.png",  
          pl.col("D").cast(pl.Int64),
          pl.col("Ms").cast(pl.Int64),
          pl.col("DMI").cast(pl.Float32),
          pl.col("Ku").cast(pl.Float32)
        ).map_elements(to_base64_resized, return_dtype=pl.String).alias("image")
      ])
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(alt, df, pl):
    filtered_df = df.filter(
            (pl.col("DMI") == 0.5) & (pl.col("Ku") == 0.08)
        )
    alt.Chart(filtered_df, title="Phase Diagram").mark_rect().encode(
        alt.X("Ms:Q", bin=True, title='Ms (Binned)').title(r"$M_s$"),
        alt.Y("D:Q", bin=True, title='Ms (Binned)').title("D"),
        alt.Color("Sk:N").title("Skyrmion"),
        tooltip=[
            alt.Tooltip("Ms", title="Ms"),
            alt.Tooltip("D", title="D"),
            alt.Tooltip("S2k_bot", title="Sk_bot", format=".3f"),
            alt.Tooltip("S2k_top", title="Sk_top", format=".3f"),
            alt.Tooltip("image")
        ]
    ).configure_view(
        step=13,
        strokeWidth=0
    ).configure_axis(
        domain=False
    ).mark_rect(stroke='white', strokeWidth=0.1)
    return


@app.cell
def _(pl, to_base64_resized):
    csv_path_sk = '../data/csv_data/saf_results_sk.csv'
    df_sk = pl.read_csv(csv_path_sk).with_columns([
        pl.format(
          r"C:\SPIN-UNI\Orlando\micromagnetics\images\saf_results_relax\bottomlayer_D={}_Ms={}_T=0_dmi={}_Ku={}.png",  
          pl.col("D").cast(pl.Int64),
          pl.col("Ms").cast(pl.Int64),
          pl.col("DMI").cast(pl.Float32),
          pl.col("Ku").cast(pl.Float32)
        ).map_elements(to_base64_resized, return_dtype=pl.String).alias("image")
      ])
    return (df_sk,)


@app.cell
def _(alt, df_sk, pl):
    filtered_df_sk = df_sk.filter(
            (pl.col("DMI") == 0.5) & (pl.col("Ku") == 0.08)
        )
    alt.Chart(filtered_df_sk, title="Phase Diagram").mark_rect().encode(
        alt.X("Ms:Q", bin=True, title='Ms (Binned)').title(r"$M_s$"),
        alt.Y("D:Q", bin=True, title='Ms (Binned)').title("D"),
        alt.Color("Sk_top").title("Skyrmion").scale(scheme="plasma"),
        tooltip=[
            alt.Tooltip("Ms", title="Ms"),
            alt.Tooltip("D", title="D"),
            alt.Tooltip("Sk_bot", title="Sk_bot", format=".3f"),
            alt.Tooltip("Sk_top", title="Sk_top", format=".3f"),
            alt.Tooltip("image")
        ]
    ).configure_view(
        step=13,
        strokeWidth=0
    ).configure_axis(
        domain=False
    ).mark_rect(stroke='white', strokeWidth=0.1)
    return


@app.cell
def _(mo):
    dmi = mo.ui.dropdown(
        options=[0.5,0.6,1.0],
        value=0.5,
        label='DMI:'
    )

    ku = mo.ui.dropdown(
        options=[x/100 for x in range(2,22,2)],
        value=0.1,
        label='Ku:'
    )
    return dmi, ku


@app.cell
def _(dmi, ku, mo):
    mo.vstack(
        [dmi,ku]
    )
    return


@app.cell
def _(dmi, ku):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    T = 0
    data_path = f'../data/saf_results_hysteresis/dmi={dmi.value}/'

    Ds = range(150, 825, 15)
    Mss = range(260, 460, 10)

    skyrmion_stability = {}

    for D in Ds:
        for Ms in Mss:
    
            filename = f'topological_charge_hyst_D={D}_Ms={Ms}_T={T}_dmi={dmi.value}_Ku={ku.value}.json'

            try:
                with open(data_path+filename) as json_file:
                    json_file = json.load(json_file)
            except:
                skyrmion_stability[(D,Ms)] = 'none'
                continue
    
            sk_values = json_file[f'({D},{Ms})']['s_k']
            s2k_values = json_file[f'({D},{Ms})']['s2_k']
            h_values = json_file[f'({D},{Ms})']['H']
            init = 0
            final = 34
            topological_charge_init = s2k_values[init][0]
            topological_charge_final = s2k_values[final][0]

            top_charge_diff = np.abs(topological_charge_final - topological_charge_init)
        
            if top_charge_diff < 1e-1 and abs(abs(s2k_values[final][0])-1) < 0.25:
            #if top_charge_diff < 1e-1:
                skyrmion_stability[(D,Ms)] = 'stable'
                #print(f'Stable for D={D}, Ms={Ms}')
            else:
                skyrmion_stability[(D,Ms)] = 'metastable'
                #print(f'Not stable for D={D}, Ms={Ms}')

    # Convert dictionary keys to arrays
    D_values = np.array([k[0] for k in skyrmion_stability.keys()])
    Ms_values = np.array([k[1] for k in skyrmion_stability.keys()])
    states = np.array([v for v in skyrmion_stability.values()])

    # Split data by state
    stable_mask = states == 'stable'
    metastable_mask = states == 'metastable'

    # Create the figure
    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot stable points (squares)
    ax.scatter(
        Ms_values[stable_mask],
        D_values[stable_mask],
        marker='o',   # square
        color='royalblue',
        s=200,
        label='Stable'
    )

    # Plot metastable points (triangles)
    ax.scatter(
        Ms_values[metastable_mask],
        D_values[metastable_mask],
        marker='x',   # triangle
        color='orange',
        s=200,
        label='Metastable'
    )

    # Labels and style
    ax.set_ylabel('D [nm]', fontsize=12)
    ax.set_xlabel('M$_s$ [kA/m]', fontsize=12)
    ax.set_title(
        rf'Stability Map for DMI={dmi.value} and $K_u$={ku.value}',
        fontsize=14,
        weight='bold'
    )

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticks(sorted(set(Ms_values)))
    ax.set_yticks(sorted(set(D_values)))

    plt.tight_layout()
    ax
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
