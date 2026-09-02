import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

    return (
        XGBClassifier,
        classification_report,
        confusion_matrix,
        f1_score,
        np,
        pl,
        precision_score,
        recall_score,
        train_test_split,
    )


@app.cell
def _(pl):
    # 1. Load Data
    csv_path = "../data/csv_data/saf_hyst_results.csv"  # Replace with actual file path

    df = pl.read_csv(csv_path).with_columns([
        pl.when(pl.col("sk_stability") == 'stable')
        .then(1)
        .otherwise(0)
        .alias("Sk")
    ])
    return (df,)


@app.cell
def _(df, np, pl):
    # 2. Feature Engineering directly in Polars
    Aexchange = 1e-11
    eps = 1e-10
    mu0 = 4 * np.pi * 1e-7

    df_feat = (
        df
        .with_columns([
            (pl.col("DMI") * 1e-3).alias("DMI_si"),
            (pl.col("Ms") * 1e3).alias("Ms_si"),
            (pl.col("Ku") * 1e6).alias("Ku_si"),
            (pl.col("D") * 1e-9).alias("D_si"),
        ])
        .with_columns([
            ((2 * pl.col("Ku_si")) / (mu0 * (pl.col("Ms_si") ** 2) + eps)).alias("Q"),
            ((Aexchange / (pl.col("Ku_si") + eps)).sqrt()).alias("lex"),
            (pl.col("DMI_si") / ((Aexchange * pl.col("Ku_si") + eps).sqrt())).alias("dmi"),
            ((np.pi * pl.col("DMI_si")) / (4 * (Aexchange * pl.col("Ku_si") + eps).sqrt())).alias("kappa"),
        ])
        .with_columns([
            (pl.col("D_si") / (pl.col("lex") + eps)).alias("D_lex_ratio")
        ])
    )
    return (df_feat,)


@app.cell
def _(df_feat):
    # Define feature columns
    feature_cols = ['D', 'Ms', 'DMI', 'Ku', 'Q', 'kappa', 'dmi', 'lex', 'D_lex_ratio']

    # 3. Convert to NumPy Arrays
    X = df_feat.select(feature_cols).to_numpy()
    y = df_feat.select("Sk").to_numpy().flatten()
    return feature_cols, y


@app.cell
def _(df_feat, feature_cols, np, train_test_split, y):
    indices = np.arange(len(df_feat))

    idx_train, idx_val, y_train, y_val = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )

    df_val = df_feat[idx_val]

    X_train = df_feat[idx_train].select(feature_cols).to_numpy()
    X_val = df_val.select(feature_cols).to_numpy()
    return X_train, X_val, df_val, y_train, y_val


@app.cell
def _(np, y_train):
    # 5. Compute Imbalance Ratio for XGBoost
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = np.sqrt(num_neg / num_pos)
    return num_neg, num_pos, scale_pos_weight


@app.cell
def _(XGBClassifier, X_train, scale_pos_weight, y_train):
    # 6. Instantiate and Train XGBoost
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    xgb.fit(X_train, y_train)
    return (xgb,)


@app.cell
def _(
    X_val,
    classification_report,
    confusion_matrix,
    f1_score,
    np,
    precision_score,
    recall_score,
    xgb,
    y_val,
):
    # 7. Evaluate Probabilities & Sweep Decision Thresholds
    val_probs = xgb.predict_proba(X_val)[:, 1]

    print(f"{'Thresh':<8} | {'Precision':<10} | {'Recall':<8} | {'F1 Score':<8}")
    print("-" * 45)

    best_f1 = 0.0
    best_thresh = 0.50

    for thresh in np.arange(0.30, 0.85, 0.05):
        preds = (val_probs >= thresh).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
    
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
        
        print(f"{thresh:<8.2f} | {prec:<10.3f} | {rec:<8.3f} | {f1:<8.3f}")

    print("\n" + "=" * 45)
    print(f"Optimal Threshold: {best_thresh:.2f} | Best Validation F1: {best_f1:.3f}")
    print("=" * 45)

    # Detailed Report at Optimal Threshold
    best_preds = (val_probs >= best_thresh).astype(int)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, best_preds))

    print("\nClassification Report:")
    print(classification_report(y_val, best_preds, target_names=['Unstable (0)', 'Stable Skyrmion (1)']))
    return (best_preds,)


@app.cell
def _(XGBClassifier, X_train, np, num_neg, num_pos, y_train):
    # Try square-root of imbalance ratio (~2.89 instead of 8.37)
    xgb_balanced = XGBClassifier(
        n_estimators=150,
        max_depth=3,                  # Reduce depth slightly to prevent boundary overfitting
        learning_rate=0.05,
        scale_pos_weight=np.sqrt(num_neg / num_pos), 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_balanced.fit(X_train, y_train)
    return


@app.cell
def _(best_preds, df_val, pl, y_val):
    # 6. Filter False Positives safely (lengths now match at 300)
    fp_mask = (y_val == 0) & (best_preds == 1)
    fp_samples = df_val.filter(pl.Series("fp", fp_mask))

    print(f"Number of False Positives: {len(fp_samples)}")
    print("\nFalse Positive Parameter Summary:")
    print(fp_samples.select(['D', 'Ms', 'DMI', 'Ku', 'Q', 'kappa', 'dmi', 'lex']).describe())
    return


@app.cell
def _(best_preds, df_val, pl, y_val):
    # Extract True Positives (Actual = 1, Predicted = 1)
    tp_mask = (y_val == 1) & (best_preds == 1)
    tp_samples = df_val.filter(pl.Series("tp", tp_mask))

    print("True Positive Parameter Summary:")
    print(tp_samples.select(['D', 'Ms', 'DMI', 'Ku', 'Q', 'kappa']).describe())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
