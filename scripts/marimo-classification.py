import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", css_file="", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    return (
        StandardScaler,
        classification_report,
        confusion_matrix,
        f1_score,
        nn,
        np,
        optim,
        pl,
        plt,
        sns,
        torch,
        train_test_split,
    )


@app.cell
def _(
    classification_report,
    confusion_matrix,
    f1_score,
    nn,
    np,
    optim,
    plt,
    sns,
):
    # ==========================================
    # 1. PHYSICS-INFORMED FEATURE ENGINEERING
    # ==========================================
    def engineer_features(data):
        """
        Input: Array of [D, Ms, DMI, Ku]
        Output: Array of [D, Ms, DMI, Ku, Q, kappa, D_reduced, Log_Ms]
        """
        mu0 = 4 * np.pi * 1e-7
        A = 10e-12  # Assuming a constant exchange stiffness (can be a variable too)

        D, Ms, DMI, Ku = data[:,0]*1e-9, data[:,1]*1e3, data[:,2]*1e-3, data[:,3]*1e6

        # Quality Factor Q
        Q = (2 * Ku) / (mu0 * Ms**2)
        # Stability Parameter Kappa
        kappa = (np.pi * DMI) / (4 * np.sqrt(A * Ku))
        # Reduced DMI
        D_red = DMI / np.sqrt(A * Ku)
        # Logarithmic Ms (to handle scale)
        log_ms = np.log10(Ms)

        extra_features = np.column_stack((Q, kappa, D_red, log_ms))
        return np.hstack((data, extra_features))

    # ==========================================
    # 2. DATA AUGMENTATION (The DenseNN_BN Method)
    # ==========================================
    def augment_magnetic_data(X, y, noise_level=0.01, copies=3):
        """Adds 'jittered' points near existing data to strengthen boundaries"""
        X_aug, y_aug = X.copy(), y.copy()
        for _ in range(copies):
            noise = np.random.normal(0, noise_level, X.shape) * X
            X_aug = np.vstack((X_aug, X + noise))
            y_aug = np.concatenate((y_aug, y))
        return X_aug, y_aug

    # ==========================================
    # 3. NEURAL NETWORK ARCHITECTURE
    # ==========================================
    class DenseNetwork_BN(nn.Module):
        def __init__(self, input_dim=8):
            super(DenseNetwork_BN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128), # Stability during training
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid() 
            )

        def forward(self, x):
            return self.net(x)

    class DenseNetwork(nn.Module):
        def __init__(self, n_features=8, dropout_rate=0.3):
            super(DenseNetwork, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate), # Stability during training
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid() 
            )

        def forward(self, x):
            return self.net(x)

    # ==========================================
    # 4. VISUALIZATION (F1 & CONFUSION MATRIX)
    # ==========================================
    def plot_model_performance(y_true, y_pred_probs):
        y_pred = (y_pred_probs >= 0.5).astype(int)

        plt.figure(figsize=(12, 5))

        # Confusion Matrix
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='magma', 
                    xticklabels=['No Skyrmion', 'Skyrmion'], 
                    yticklabels=['No Skyrmion', 'Skyrmion'])
        plt.title("Phase Prediction Confusion Matrix")

        # F1 Score Bar Chart (Comparison Style)
        plt.subplot(1, 2, 2)
        f1 = f1_score(y_true, y_pred)
        plt.bar(['Model F1-Score'], [f1], color='teal')
        plt.ylim(0, 1)
        plt.title(f"F1 Accuracy: {f1:.4f}")

        plt.tight_layout()
        plt.show()
        print(classification_report(y_true, y_pred))

    # ==========================================
    # 2. COMPARISON & VISUALIZATION FUNCTION
    # ==========================================
    def compare_models(y_true, pred_dnn_bn, pred_dense):
        # Convert probabilities to binary 0 or 1
        y_dnn_bn = (pred_dnn_bn >= 0.5).astype(int)
        y_dense = (pred_dense >= 0.5).astype(int)

        f1_dnn_bn = f1_score(y_true, y_dnn_bn)
        f1_dense = f1_score(y_true, y_dense)

        plt.figure(figsize=(15, 5))

        # Confusion Matrix: DenseNN_BN
        plt.subplot(1, 3, 1)
        sns.heatmap(confusion_matrix(y_true, y_dnn_bn), annot=True, fmt='d', cmap='Blues')
        plt.title(f"DenseNN_BN-Style\nF1: {f1_dnn_bn:.3f}")
        plt.ylabel('Actual'); plt.xlabel('Predicted')

        # Confusion Matrix: DenseNetwork
        plt.subplot(1, 3, 2)
        sns.heatmap(confusion_matrix(y_true, y_dense), annot=True, fmt='d', cmap='Greens')
        plt.title(f"DenseNetwork\nF1: {f1_dense:.3f}")
        plt.ylabel('Actual'); plt.xlabel('Predicted')

        # F1 Score Comparison Bar Chart
        plt.subplot(1, 3, 3)
        models = ['DenseNN_BN', 'DenseNet']
        scores = [f1_dnn_bn, f1_dense]
        plt.bar(models, scores, color=['blue', 'green'])
        plt.ylim(0, 1)
        plt.title("F1-Score Comparison")

        plt.tight_layout()
        plt.show()

    def train_model(model, X_train, y_train, epochs=100):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
        return model

    return (
        DenseNetwork,
        DenseNetwork_BN,
        augment_magnetic_data,
        compare_models,
        engineer_features,
        train_model,
    )


@app.cell
def _(pl):
    df = pl.read_csv("../data/csv_data/saf_relax-results.csv").with_columns([
        pl.when(abs(pl.col("S2k_bot") - 1) < 0.3)
            .then(1)
            .otherwise(0)
            .alias("skyrmion_bool"),
        pl.format(
            "https://github.com/OrlandoDomo/micromagnetics/raw/main/images/saf_skyrmion-0.8nm-z_0.8nm-results/skyrmion-{}nm-{}kA_m.png",
            pl.col("D").cast(pl.Int64),
            pl.col("Ms").cast(pl.Int64)
        ).alias("image")
    ])
    return (df,)


@app.cell
def _(df):
    X_raw = df['D','Ms','DMI','Ku'].to_numpy()
    y = df['skyrmion_bool'].to_numpy()
    return X_raw, y


@app.cell
def _(X_raw, engineer_features):
    # Step 1: Feature Engineering
    X_engineered = engineer_features(X_raw)
    return (X_engineered,)


@app.cell
def _(X_engineered, augment_magnetic_data, train_test_split, y):
    # Step 2: Augmentation
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2)
    X_train_aug, y_train_aug = augment_magnetic_data(X_train, y_train)
    return X_test, X_train_aug, y_test, y_train, y_train_aug


@app.cell
def _(X_train_aug, pl):
    pl.DataFrame(X_train_aug, schema=['D','Ms','DMI','Ku','Q','k','dmi','Ms_log']).sample(10)
    return


@app.cell
def _(StandardScaler, X_test, X_train_aug, torch, y_train_aug):
    # Step 3: Scaling (Crucial for Neural Nets)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)

    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train_aug).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test_scaled)
    return X_test_t, X_train_t, scaler, y_train_t


@app.cell
def _(DenseNetwork, DenseNetwork_BN, X_train_t, train_model, y_train_t):
    # Initialize
    model_dnn_bn = DenseNetwork_BN(input_dim=8)
    model_dense = DenseNetwork(n_features=8)

    # Train both
    print("Training DenseNN_BN model...")
    model_dnn_bn = train_model(model_dnn_bn, X_train_t, y_train_t)
    print("Training DenseNetwork...")
    model_dense = train_model(model_dense, X_train_t, y_train_t)
    return model_dense, model_dnn_bn


@app.cell
def _(X_test_t, compare_models, model_dense, model_dnn_bn, torch, y_test):
    # Evaluate
    model_dnn_bn.eval()
    model_dense.eval()
    with torch.no_grad():
        res_dnn_bn = model_dnn_bn(X_test_t).numpy()
        res_dense = model_dense(X_test_t).numpy()

    compare_models(y_test, res_dnn_bn, res_dense)
    return


@app.cell
def _(confusion_matrix, f1_score, nn, optim, plt, sns, torch):
    # ==========================================
    # 1. UPDATED TRAINING LOOP (WITH HISTORY)
    # ==========================================
    def train_with_history(model, X_train, y_train, X_val, y_val, epochs=150):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            # Track validation loss (to see overfitting)
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

        return model, train_losses, val_losses

    # ==========================================
    # 2. FULL DIAGNOSTIC VISUALIZATION
    # ==========================================
    def plot_full_diagnostic(y_true, res_dnn_bn, res_dense, hist_dnn_bn, hist_dense):
        y_dnn_bn = (res_dnn_bn >= 0.5).astype(int)
        y_dense = (res_dense >= 0.5).astype(int)

        fig = plt.figure(figsize=(16, 10))

        # Row 1: Loss Curves (The "Learning" Progress)
        plt.subplot(2, 2, 1)
        plt.plot(hist_dnn_bn[0], label='DenseNN_BN Train', color='blue', linestyle='--')
        plt.plot(hist_dnn_bn[1], label='DenseNN_BN Val', color='blue')
        plt.title("DenseNN_BN Loss (BatchNorm)")
        plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(hist_dense[0], label='DenseNet Train', color='green', linestyle='--')
        plt.plot(hist_dense[1], label='DenseNet Val', color='green')
        plt.title("DenseNet Loss (Dropout)")
        plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()

        # Row 2: Confusion Matrices
        plt.subplot(2, 2, 3)
        sns.heatmap(confusion_matrix(y_true, y_dnn_bn), annot=True, fmt='d', cmap='Blues')
        plt.title(f"DenseNN_BN Confusion Matrix\nF1: {f1_score(y_true, y_dnn_bn):.3f}")

        plt.subplot(2, 2, 4)
        sns.heatmap(confusion_matrix(y_true, y_dense), annot=True, fmt='d', cmap='Greens')
        plt.title(f"DenseNet Confusion Matrix\nF1: {f1_score(y_true, y_dense):.3f}")

        plt.tight_layout()
        plt.show()

    return plot_full_diagnostic, train_with_history


@app.cell
def _(y_train):
    num_pos = (y_train == 1).sum().item()
    num_neg = (y_train == 0).sum().item()
    pos_weight = num_neg / num_pos
    pos_weight
    return


@app.cell
def _(
    DenseNetwork,
    DenseNetwork_BN,
    X_test_t,
    X_train_t,
    plot_full_diagnostic,
    torch,
    train_with_history,
    y_test,
    y_train_t,
):
    # --- Execution ---
    model_dnn_bn_new = DenseNetwork_BN(input_dim=8)
    model_dense_new = DenseNetwork(n_features=8)

    print("Training Models...")
    model_dnn_bn_hist, train_f, val_f = train_with_history(model_dnn_bn_new, X_train_t, y_train_t, X_test_t, torch.FloatTensor(y_test).view(-1,1))
    model_dense_hist, train_d, val_d = train_with_history(model_dense_new, X_train_t, y_train_t, X_test_t, torch.FloatTensor(y_test).view(-1,1))

    # Final Evaluation
    model_dnn_bn_hist.eval()
    model_dense_hist.eval()
    with torch.no_grad():
        final_dnn_bn = model_dnn_bn_hist(X_test_t).numpy()
        final_dense = model_dense_hist(X_test_t).numpy()

    plot_full_diagnostic(y_test, final_dnn_bn, final_dense, (train_f, val_f), (train_d, val_d))
    return model_dense_hist, model_dnn_bn_hist


@app.cell
def _(confusion_matrix, engineer_features, f1_score, plt, sns, torch):
    def compare_new_data(models_dictionary, raw_data, true_labels, scaler):
        # 1. Apply the same Physics-Informed Engineering (4 -> 8 columns)
        X_engineered = engineer_features(raw_data)

        # 2. Scale using the ORIGINAL scaler (Crucial: do not fit a new one!)
        X_scaled = scaler.transform(X_engineered)
        X_tensor = torch.FloatTensor(X_scaled)

        results = {}

        plt.figure(figsize=(6 * len(models_dictionary), 5))

        for i, (name, model) in enumerate(models_dictionary.items()):
            model.eval()
            with torch.no_grad():
                # Get raw probabilities
                probs = model(X_tensor).numpy().flatten()
                preds = (probs >= 0.5).astype(int)

                correct = torch.eq(torch.Tensor(true_labels), torch.Tensor(preds)).sum().item()
                acc = (correct / len(preds)) * 100 
                print(f"Accuracy for model {name} is {acc:.2f}")

                f1 = f1_score(true_labels, preds)
                results[name] = f1

                # Plotting the Confusion Matrix for this specific dataset
                plt.subplot(1, len(models_dictionary), i + 1)
                cm = confusion_matrix(true_labels, preds)
                sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
                plt.title(f"Model: {name}\nUnseen Data F1: {f1:.3f}")
                plt.ylabel('Actual'); plt.xlabel('Predicted')

        plt.tight_layout()
        plt.show()

        return results

    return (compare_new_data,)


@app.cell
def _(pl):
    pl.concat(
        [
            pl.read_csv("../data/csv_data/saf_relax-dmi=0.8_ku=0.08.csv").with_columns([
                pl.when(abs(pl.col("S2k_bot") - 1) < 0.3)
                    .then(1)
                    .otherwise(0)
                    .alias("skyrmion_bool")
            ]),
            pl.read_csv("../data/csv_data/saf_relax-dmi=0.7_ku=0.08.csv").with_columns([
                pl.when(abs(pl.col("S2k_bot") - 1) < 0.3)
                    .then(1)
                    .otherwise(0)
                    .alias("skyrmion_bool")
            ])
        ]
    ).write_csv("../data/csv_data/saf_relax-dmi=0.6-8_ku=0.08.csv")
    return


@app.cell
def _(pl):
    df_new = pl.read_csv("../data/csv_data/saf_relax-hi_res.csv").with_columns([
        pl.when(abs(pl.col("S2k_bot") - 1) < 0.3)
            .then(1)
            .otherwise(0)
            .alias("skyrmion_bool")
    ])

    x_new = df_new['D','Ms','DMI','Ku'].to_numpy()
    y_new = df_new['skyrmion_bool'].to_numpy()
    return x_new, y_new


@app.cell
def _(
    compare_new_data,
    model_dense_hist,
    model_dnn_bn_hist,
    scaler,
    x_new,
    y_new,
):
    my_models = {
         "DenseNN_BN_Final": model_dnn_bn_hist,
         "DenseNet_Final": model_dense_hist
    }

    scores = compare_new_data(my_models, x_new, y_new, scaler)
    print("Final Comparison Scores on Unseen Data:", scores)
    return


@app.cell(disabled=True)
def _(model_dnn_bn_hist, scaler, torch):
    torch.save({
        'model_state_dict': model_dnn_bn_hist.state_dict(),
        'scaler': scaler,
        'model_type': 'Dense BatchNorm',
        },
        fr"ml\saved_models\dense_bnn.pt"
    )
    return


@app.cell
def _(ListedColormap, np, plt, predict_single):
    def create_phase_diagram(model, scaler, model_type, grid_dims, DMI, K, 
                              D_range=(150, 825), Ms_range=(260, 460), 
                              resolution=100, device='cpu', save_path=None):
      # Create grid - Ms on x-axis, D on y-axis
      Ms_values = np.linspace(Ms_range[0], Ms_range[1], resolution)
      D_values = np.linspace(D_range[0], D_range[1], resolution)
      Ms_grid, D_grid = np.meshgrid(Ms_values, D_values)

      # Predict for each point
      predictions = np.zeros((resolution, resolution))
      probabilities = np.zeros((resolution, resolution))

      print(f"Generating phase diagram for DMI={DMI}, K={K}...")
      print(f"Ms range (x-axis): {Ms_range[0]} to {Ms_range[1]}")
      print(f"D range (y-axis): {D_range[0]} to {D_range[1]}")
      print(f"Resolution: {resolution}x{resolution} = {resolution**2} points")

      for i in range(resolution):
        for j in range(resolution):
          D = D_grid[i, j]
          Ms = Ms_grid[i, j]
          pred, prob = predict_single(model, scaler, model_type, grid_dims, D, Ms, DMI, K, device)
          predictions[i, j] = pred
          probabilities[i, j] = prob

        if (i + 1) % 10 == 0:
          print(f"Progress: {i+1}/{resolution} rows completed")

      # Create figure
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

      # Plot 1: Binary predictions
      cmap_binary = ListedColormap(['blue', 'red'])
      im1 = ax1.imshow(predictions, extent=[Ms_range[0], Ms_range[1], D_range[0], D_range[1]],
                       origin='lower', cmap=cmap_binary, aspect='auto', interpolation='nearest')
      ax1.set_xlabel('Ms', fontsize=12)
      ax1.set_ylabel('D', fontsize=12)
      ax1.set_title(f'Phase Diagram (DMI={DMI}, K={K})\nBinary Prediction', fontsize=14)
      cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1])
      cbar1.set_label('Sk', fontsize=12)
      ax1.grid(True, alpha=0.3)

      plt.tight_layout()

      return fig

    return


if __name__ == "__main__":
    app.run()
