import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from ml.predicting import create_phase_diagram, load_model
    from matplotlib.colors import ListedColormap
    return (
        ListedColormap,
        create_phase_diagram,
        load_model,
        mo,
        np,
        pd,
        plt,
        torch,
    )


@app.cell
def _(mo):
    model_type_path = mo.ui.dropdown(
        options={
            "Dense NN": 'dense',
            "2D CNN Separate": 'option1',
            "2D CNN 4Channels": 'option2',
            "3D CNN": 'option3',
        },
        value='Dense NN',
        label='ML Model:'
    )
    return (model_type_path,)


@app.cell
def _(model_type_path):
    model_type_path
    return


@app.cell
def _(model_type_path):
    model_path = f"saved_models/{model_type_path.value}_model.pt"
    return (model_path,)


@app.cell
def _(load_model, model_path, torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    print(f"Loading model from {model_path}...")
    model, scaler, model_type, grid_dims = load_model(model_path, device)
    print(f"Model type: {model_type}")
    print(f"Grid dimensions: D={grid_dims['n_d']}, Ms={grid_dims['n_ms']}, "
    f"DMI={grid_dims['n_dmi']}, K={grid_dims['n_k']}")
    return device, grid_dims, model, model_type, scaler


@app.cell
def _():
    D_min = 150
    D_max = 825
    Ms_min = 260
    Ms_max = 460
    #resolution = 20

    #save_path = f'figures/{model_type}-predict-dmi={DMI}-k={Ku}.png'
    return D_max, D_min, Ms_max, Ms_min


@app.cell
def _(mo):
    DMI = mo.ui.dropdown(
        options=[x/10 for x in range(1,21,1)],
        value=0.2,
        label='DMI:'
    )

    Ku = mo.ui.dropdown(
        options=[x/100 for x in range(1,21,1)],
        value=0.02,
        label='Ku:'
    )

    resolution = mo.ui.dropdown(
        options=[x for x in range(1,101,1)],
        value=10,
        label='Resolution:'
    )
    return DMI, Ku, resolution


@app.cell
def _(resolution):
    resolution
    return


@app.cell
def _(DMI):
    DMI
    return


@app.cell
def _(Ku):
    Ku
    return


@app.cell
def _(
    DMI,
    D_max,
    D_min,
    Ku,
    Ms_max,
    Ms_min,
    create_phase_diagram,
    device,
    grid_dims,
    model,
    model_type,
    resolution,
    scaler,
):
    create_phase_diagram(
        model, scaler, model_type, grid_dims,
        DMI=DMI.value, K=Ku.value,
        D_range=(D_min, D_max),
        Ms_range=(Ms_min, Ms_max),
        resolution=resolution.value,
        device=device,
        save_path=None
    )
    return


@app.cell
def _(mo):
    simulation_dmi = mo.ui.dropdown(
        options=[0.5,1.0],
        value=0.5,
        label='Simulated DMI:'
    )

    simulation_ku = mo.ui.dropdown(
        options=[x/100 for x in range(0,22,2)],
        value=0.02,
        label='Simulated Ku:'
    )

    tolerance = mo.ui.dropdown(
        options=[x/10 for x in range(1,4)],
        value=0.1,
        label='Tolerance:'
    )
    return simulation_dmi, simulation_ku, tolerance


@app.cell
def _(simulation_dmi):
    simulation_dmi
    return


@app.cell
def _(simulation_ku):
    simulation_ku
    return


@app.cell
def _(tolerance):
    tolerance
    return


@app.cell
def _(np, pd, simulation_dmi, simulation_ku, tolerance):
    dmi_value = simulation_dmi.value
    ku_value = simulation_ku.value
    df = pd.read_csv("data\data\saf_skyrmion_results_final.csv").query('DMI==@dmi_value and Ku==@ku_value')
    df['Sk'] = (np.abs(df['S2k_bot'] - 1) < tolerance.value).astype(int)
    #df.query('DMI==1 and Ku==0.04')
    #df[(df.DMI==1.0)&(df.Ku==0.04)]
    return df, dmi_value, ku_value


@app.cell
def _(
    D_max,
    D_min,
    ListedColormap,
    Ms_max,
    Ms_min,
    df,
    dmi_value,
    ku_value,
    plt,
):
    Ms_values = range(Ms_min,Ms_max,20)
    D_values = range(D_min,D_max,75)
    #Ms_grid, D_grid = np.meshgrid(Ms_values, D_values)
    grid = df.pivot(index='D', columns='Ms', values='Sk')

    cmap_binary = ListedColormap(['blue', 'red'])
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    simulation_im = ax.imshow(grid.values,
        aspect='auto',
        origin='lower',
        extent=[grid.columns.min(), grid.columns.max(),
               grid.index.min(), grid.index.max()],
        cmap=cmap_binary,
        vmin=0,
        vmax=1
    )
    ax.set_xlabel(r"$M_s$ [kA/m]")
    ax.set_ylabel(r"$D$ [nm]")
    ax.set_title(f"Phase Diagram (DMI={dmi_value}, K={ku_value})")
    cbar = plt.colorbar(simulation_im, ax=ax, ticks=[0, 1])
    cbar.set_label('Sk', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
