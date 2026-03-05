import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import os
    from ml.predicting import create_phase_diagram, load_model, predict_single
    from matplotlib.colors import ListedColormap

    return (
        ListedColormap,
        create_phase_diagram,
        load_model,
        mo,
        np,
        os,
        pd,
        plt,
        predict_single,
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
    model_path = f"ml/saved_models/{model_type_path.value}_model.pt"
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
    D_max = 750
    Ms_min = 260
    Ms_max = 440
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
def _(DMI, Ku, mo, resolution):
    mo.vstack([resolution,DMI,Ku])
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
def _(mo, os):
    full_path = r'O:\UNI\maestria_2025\micromagnetics\notes\data\csv_data'
    files = os.listdir(full_path)
    csv_path_options = mo.ui.dropdown(
        [x for x in files if '.csv' in x],
        label='Choose CSV File:'
    )
    return csv_path_options, full_path


@app.cell
def _(csv_path_options):
    csv_path_options
    return


@app.cell
def _(mo):
    tolerance = mo.ui.dropdown(
        options=[x/10 for x in range(1,4)],
        value=0.1,
        label='Tolerance:'
    )
    return (tolerance,)


@app.cell
def _(tolerance):
    tolerance
    return


@app.cell
def _(csv_path_options, full_path, np, pd, tolerance):
    #df = pd.read_csv(f'{full_path}/{csv_path_options.value}').query('DMI==@dmi_value and Ku==@ku_value')
    df = pd.read_csv(f'{full_path}/{csv_path_options.value}')
    df['Sk'] = (np.abs(df['S2k_bot'] - 1) < tolerance.value).astype(int)
    #df.query('DMI==1 and Ku==0.04')
    #df[(df.DMI==1.0)&(df.Ku==0.04)]
    return (df,)


@app.cell
def _(df, mo):
    simulation_dmi_options = [float(x) for x in list(set(df.DMI.values))]
    simulation_dmi = mo.ui.dropdown(
        options=simulation_dmi_options,
        value=simulation_dmi_options[0],
        label='Simulated DMI:'
    )

    simulation_ku_options = [float(x) for x in list(set(df.Ku.values))]
    simulation_ku = mo.ui.dropdown(
        #options=[x/100 for x in range(0,22,2)],
        options=simulation_ku_options,
        value=simulation_ku_options[0],
        label='Simulated Ku:'
    )
    return simulation_dmi, simulation_ku


@app.cell
def _(mo, simulation_dmi, simulation_ku):
    mo.vstack([simulation_dmi, simulation_ku])
    return


@app.cell
def _(simulation_dmi, simulation_ku):
    dmi_value = simulation_dmi.value
    ku_value = simulation_ku.value
    return dmi_value, ku_value


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
    filtered_df = df.query('DMI==@dmi_value and Ku==@ku_value').reset_index()
    grid = filtered_df.pivot(index='D', columns='Ms', values='Sk')

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
    cbar.set_ticklabels(ticklabels=['Other','Skyrmion'])
    cbar.set_label('Skyrmion', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig
    return (filtered_df,)


@app.cell
def _(
    device,
    filtered_df,
    grid_dims,
    model,
    model_type,
    predict_single,
    scaler,
):
    predictions = []
    for i in range(len(filtered_df)):

        prediction, prob = predict_single(
            model=model,
            scaler=scaler,
            model_type=model_type,
            grid_dims=grid_dims,
            D=filtered_df.D[i],
            Ms=filtered_df.Ms[i],
            DMI=filtered_df.DMI[i],
            K=filtered_df.Ku[i],
            device=device
        )

        predictions.append(prediction)
    return (predictions,)


@app.cell
def _(filtered_df, mo, predictions):
    filtered_df['s2k_predicted'] = predictions
    filtered_df['prediction_diff'] = abs(filtered_df['Sk'] - filtered_df['s2k_predicted'])
    filtered_df['skyrmion_prediction_acc'] = (filtered_df['prediction_diff'] == 0) & (filtered_df['s2k_predicted'] == 1)
    correct_predict = (filtered_df['skyrmion_prediction_acc'] == 1).values.sum()
    percent_correct = correct_predict/((filtered_df['Sk']==1).values.sum())
    mo.md(f'Accuracy: {percent_correct:.0%}')
    return


@app.cell
def _(filtered_df):
    filtered_df
    return


@app.cell
def _(ListedColormap, dmi_value, filtered_df, ku_value, plt):
    grid_diff = filtered_df.pivot(index='D', columns='Ms', values='skyrmion_prediction_acc')

    cmap_binary_diff = ListedColormap(['blue', 'green'])
    fig_diff, ax_diff = plt.subplots(1, 1, figsize=(14, 6))

    simulation_im_diff = ax_diff.imshow(grid_diff.values,
        aspect='auto',
        origin='lower',
        extent=[grid_diff.columns.min(), grid_diff.columns.max(),
               grid_diff.index.min(), grid_diff.index.max()],
        cmap=cmap_binary_diff,
        vmin=0,
        vmax=1
    )
    ax_diff.set_xlabel(r"$M_s$ [kA/m]")
    ax_diff.set_ylabel(r"$D$ [nm]")
    ax_diff.set_title(f"Phase Diagram difference(DMI={dmi_value}, K={ku_value})")
    cbar_diff = plt.colorbar(simulation_im_diff, ax=ax_diff, ticks=[0,1])
    cbar_diff.set_ticklabels(ticklabels=['Incorrect','Correct'])
    cbar_diff.set_label('Correct', fontsize=12)
    ax_diff.grid(True, alpha=0.3)
    fig_diff
    return


if __name__ == "__main__":
    app.run()
