import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import re

    return mo, np, pl, plt


@app.cell
def _(np, pl):
    data_dir_path = '../data/saf_results_relax/'
    glob_path = f"{data_dir_path}/*.txt"

    regex_pattern = r'=([+-]?\d+(?:\.\d+)?)'

    energy_cols = [
        'E_total (J)', 'E_anis (J)', 'E_demag (J)', 
        'E_exch (J)', 'E_therm (J)', 'E_custom (J)'
    ]
    sub_energies = [
        'E_anis (J)', 'E_demag (J)', 'E_exch (J)', 
        'E_therm (J)', 'E_custom (J)'
    ]
    d_conversion_factor = 1e-9
    df = (
        pl.scan_csv(glob_path, separator='\t', include_file_paths="file_path")
        .select(
            # Create the new parameter columns
            pl.col("file_path").str.extract(r'D' + regex_pattern).cast(pl.Float64).alias("D"),
            pl.col("file_path").str.extract(r'Ms' + regex_pattern).cast(pl.Float64).alias("Ms"),
            pl.col("file_path").str.extract(r'T' + regex_pattern).cast(pl.Float64).alias("T"),
            pl.col("file_path").str.extract(r'dmi' + regex_pattern).cast(pl.Float64).alias("dmi"),
            pl.col("file_path").str.extract(r'Ku' + regex_pattern).cast(pl.Float64).alias("Ku"),
            pl.col(energy_cols) 
        )
        .with_columns(
            # 3. Calculate E_dmi (J) 
            (
                pl.col('E_total (J)') - pl.sum_horizontal(sub_energies)
            ).alias('E_dmi (J)')
        )
        .with_columns(
            # Calculate Area and divide to get J/m^2
            # Formula: E_dmi / (pi * (D * conversion)^2 / 4)
            (pl.col('E_dmi (J)') / (np.pi * ((pl.col('D') * d_conversion_factor) ** 2) / 4)).alias('E_dmi (J/m^2)'),
            (pl.col('E_custom (J)') / (np.pi * ((pl.col('D') * d_conversion_factor) ** 2) / 4)).alias('E_custom (J/m^2)'),
            (pl.col('E_demag (J)') / (3 * d_conversion_factor * np.pi * ((pl.col('D') * d_conversion_factor) ** 2) / 4)).alias('E_demag (J/m^3)'),
        )
        .collect()
    )
    energy_cols.extend(['E_dmi (J)', 'E_dmi (J/m^2)', 'E_custom (J/m^2)', 'E_demag (J/m^3)'])
    return df, energy_cols


@app.cell
def _(energy_cols, mo):
    energy = mo.ui.dropdown(
        options = energy_cols,
        value='E_total (J)'
    )
    energy
    return (energy,)


@app.cell
def _(df, energy, pl, plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    for d in df['D'].unique().to_list():
        result = df.filter(pl.col('D')==d)
        ax1.plot(result['Ms'], result[energy.value], label=f'D={d:.0f} nm', marker='o')

    ax1.set_xlabel('Ms')
    ax1.set_ylabel(energy.value)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    for ms in df['Ms'].unique().to_list():
        result2 = df.filter(pl.col('Ms')==ms)
        ax2.plot(result2['D'], result2[energy.value], label=f'Ms={ms:.0f} kA/m', marker='o')

    ax2.set_xlabel('D')
    ax2.set_ylabel(energy.value)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np, pl, plt):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    def plot_pm(csv_filename="../data\csv_data\saf_relax-hi_res.csv"):

        df = pl.read_csv(csv_filename)

        dmi = 0.5
        ku = 0.08
        df = df.filter((pl.col('DMI') == dmi) & (pl.col('Ku') == ku))

        df = df.with_columns(
            pl.when(pl.col('S2k_bot') < 0.75).then(2)
            .when((pl.col('S2k_bot') >= 0.75) & (pl.col('S2k_bot') <= 1.25)).then(1)
            .when((pl.col('S2k_bot') > 1.25) & (pl.col('S2k_bot') <= 1.9)).then(3)
            .when((pl.col('S2k_bot') > 1.9) & (pl.col('S2k_bot') <= 2.1)).then(4)
            .when(pl.col('S2k_bot') >= 2.2).then(5)    
            .otherwise(6)
            .alias('Region')
        )

        grid_data = df.pivot(
            values='Region', 
            index='D', 
            on='Ms',
            aggregate_function='first'
        ).sort('D')

        # Extract the Y-axis (D) directly to a NumPy array
        D_grid = grid_data['D'].to_numpy()

        # Extract the X-axis (Ms) from the column names (excluding the 'D' column)
        ms_cols = [col for col in grid_data.columns if col != 'D']
        Ms_grid = np.array([float(col) for col in ms_cols])

        # Extract the region data columns into a 2D NumPy array
        Phase_grid = grid_data.select(ms_cols).to_numpy()

        colors = [
            '#000080',  # Region 1: Navy
            '#FFFF00',  # Region 2: Yellow
            '#006400',  # Region 3: Dark Green
            '#FF0000',  # Region 4: Red
            '#9400D3',   # Region 5: Purple
        ]
        cmap_custom = ListedColormap(colors)
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = BoundaryNorm(bounds, cmap_custom.N)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot with Ms on X and D on Y
        mesh = ax.pcolormesh(Ms_grid, D_grid, Phase_grid, 
                             cmap=cmap_custom, norm=norm, shading='nearest')

        region_labels = [
            'Skyrmion(0.75 \u2264 S2k \u2264 1.25)', 
            'Uniforme (S2k < 0.75)', 
            'Complejo (1.25 < S2k \u2264 1.9)', 
            'Skyrmionium (1.9 < S2k \u2264 2.1)', 
            'Laberinto (S2k > 2.1)'
        ]
        # Create the colored boxes (patches) for the legend
        legend_patches = [
            mpatches.Patch(color=colors[i], label=region_labels[i]) 
            for i in range(len(colors))
        ]

        # Add the legend. bbox_to_anchor pushes it outside the right edge of the plot.
        ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  fontsize=12, frameon=True, edgecolor='black')

        # Format the axes (Swapped labels)
        ax.set_xlabel('Ms (kA/m)', fontsize=16)
        ax.set_ylabel('D (nm)', fontsize=16)
        ax.tick_params(direction='in', length=6, width=1, top=True, right=True, labelsize=12)

        # Ensure the plot perfectly hugs the boundaries of your sweep
        ax.set_xlim(Ms_grid.min(), Ms_grid.max())
        ax.set_ylim(D_grid.min(), D_grid.max())

        ax.set_title("Diagrama de Fases\n" + rf"DMI={dmi} mJ/m$^2$, $K_u$={ku} MJ/m$^3$")
        plt.tight_layout()
        plt.show()

    return (plot_pm,)


@app.cell
def _(plot_pm):
    plot_pm()
    return


@app.cell
def _(pl):
    pl.read_csv("../data\csv_data\saf_relax-hi_res.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
