import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import os
    import re
    import glob
    import matplotlib
    import warnings
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torchvision.transforms as T

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    return (
        DataLoader,
        Dataset,
        Image,
        KMeans,
        PCA,
        T,
        glob,
        nn,
        np,
        os,
        pd,
        plt,
        re,
        silhouette_score,
        torch,
        warnings,
    )


@app.cell
def _(np, os, re, warnings):
    PARAM_PATTERN = re.compile(
        r"D=(?P<D>-?[\d.]+)_Ms=(?P<Ms>-?[\d.]+)_T=(?P<T>-?[\d.]+)_"
        r"dmi=(?P<dmi>-?[\d.]+)_Ku=(?P<Ku>-?[\d.]+)_Sk=(?P<Sk>-?[\d.]+)"
    )

    def parse_params(filename: str) -> dict:
        """Extract the physical parameters embedded in a filename.

        Falls back to NaNs (with a warning) if a filename doesn't match the
        expected pattern, so a single oddly-named file doesn't crash the run.
        """
        match = PARAM_PATTERN.search(os.path.basename(filename))
        if match is None:
            warnings.warn(f"Could not parse parameters from filename: {filename}")
            return {k: np.nan for k in ["D", "Ms", "T", "dmi", "Ku", "Sk"]}
        return {k: float(v) for k, v in match.groupdict().items()}

    return (parse_params,)


@app.cell
def _(Dataset, Image, T):
    class SAFImageDataset(Dataset):
        def __init__(self, filepaths, img_size=128, augment=False, grayscale=False):
            self.filepaths = filepaths
            self.grayscale = grayscale

            tfms = [T.Resize((img_size, img_size))]
            if augment:
                tfms += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
            tfms += [T.ToTensor()]  # scales to [0, 1]
            self.transform = T.Compose(tfms)

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            path = self.filepaths[idx]
            img = Image.open(path).convert("L" if self.grayscale else "RGB")
            img = self.transform(img)
            return img, path

    return (SAFImageDataset,)


@app.cell
def _(nn):
    class ConvAutoencoder(nn.Module):
        def __init__(self, in_channels=1, latent_dim=32, img_size=128):
            super().__init__()
            self.img_size = img_size

            # Encoder: 4 downsampling blocks -> img_size / 16
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
            self.reduced_size = img_size // 16
            flat_dim = 64 * self.reduced_size * self.reduced_size

            self.to_latent = nn.Linear(flat_dim, latent_dim)
            self.from_latent = nn.Linear(latent_dim, flat_dim)

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.encoder_conv(x)
            h = h.flatten(1)
            z = self.to_latent(h)
            return z

        def decode(self, z):
            h = self.from_latent(z)
            h = h.view(-1, 64, self.reduced_size, self.reduced_size)
            return self.decoder_conv(h)

        def forward(self, x):
            z = self.encode(x)
            recon = self.decode(z)
            return recon, z

    return (ConvAutoencoder,)


@app.cell
def _(nn, torch):
    def train_autoencoder(model, loader, epochs, lr, device):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for imgs, _ in loader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * imgs.size(0)
            avg_loss = total_loss / len(loader.dataset)
            if epoch == 1 or epoch % max(1, epochs // 20) == 0 or epoch == epochs:
                print(f"Epoch {epoch:4d}/{epochs}  |  reconstruction MSE: {avg_loss:.5f}")
        return model

    return (train_autoencoder,)


@app.cell
def _(DataLoader, Image, SAFImageDataset, T, nn, np, torch):
    @torch.no_grad()
    def extract_embeddings(model, filepaths, img_size, device, grayscale=True):
        """Run the (non-augmented) dataset through the trained encoder."""
        model.eval()
        dataset = SAFImageDataset(filepaths, img_size=img_size, augment=False, grayscale=grayscale)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        embeddings, ordered_paths = [], []
        for imgs, paths in loader:
            imgs = imgs.to(device)
            z = model.encode(imgs)
            embeddings.append(z.cpu().numpy())
            ordered_paths.extend(paths)
        return np.concatenate(embeddings, axis=0), ordered_paths

    @torch.no_grad()
    def extract_pretrained_embeddings(filepaths, img_size, device):
        """Alternative feature extractor: frozen ImageNet ResNet18 (no training)."""
        import torchvision.models as models

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        resnet.to(device).eval()

        tfm = T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        embeddings, ordered_paths = [], []
        for path in filepaths:
            img = Image.open(path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            feat = resnet(x).cpu().numpy().squeeze(0)
            embeddings.append(feat)
            ordered_paths.append(path)
        return np.stack(embeddings, axis=0), ordered_paths

    return extract_embeddings, extract_pretrained_embeddings


@app.cell
def _(KMeans, silhouette_score):
    def choose_k_and_cluster(embeddings, k_min, k_max, fixed_k, seed):
        if fixed_k is not None:
            km = KMeans(n_clusters=fixed_k, n_init=10, random_state=seed).fit(embeddings)
            return km, fixed_k, {}

        scores = {}
        best_k, best_km, best_score = None, None, -1
        for k in range(k_min, k_max + 1):
            if k >= len(embeddings):
                break
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(embeddings)
            score = silhouette_score(embeddings, km.labels_)
            scores[k] = score
            print(f"k={k:2d}  silhouette={score:.4f}")
            if score > best_score:
                best_k, best_km, best_score = k, km, score
        return best_km, best_k, scores

    return (choose_k_and_cluster,)


@app.cell
def _(Image, PCA, np, plt):
    def plot_embeddings_2d(embeddings, labels, out_path):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)

        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=60, edgecolor="k")
        plt.legend(*scatter.legend_elements(), title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        plt.title("Image embeddings (PCA projection), colored by cluster")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


    def plot_cluster_grid(df, out_path, n_per_cluster=6):
        clusters = sorted(df["cluster"].unique())
        fig, axes = plt.subplots(len(clusters), n_per_cluster,
                                  figsize=(2 * n_per_cluster, 2 * len(clusters)))
        if len(clusters) == 1:
            axes = axes[np.newaxis, :]

        for row, c in enumerate(clusters):
            subset = df[df["cluster"] == c].sample(
                n=min(n_per_cluster, (df["cluster"] == c).sum()), random_state=0
            )
            for col in range(n_per_cluster):
                ax = axes[row, col]
                ax.axis("off")
                if col < len(subset):
                    path = subset.iloc[col]["filepath"]
                    img = Image.open(path).convert("RGB")
                    ax.imshow(img)
                    if col == 0:
                        ax.set_title(f"cluster {c}", loc="left", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    return plot_cluster_grid, plot_embeddings_2d


@app.cell
def _():
    seed = 123
    out_dir = '../data/unsupervised-test'
    data_dir = '../images/saf_results_relax'
    feature_extractor = "autoencoder"
    k_min = 4
    k_max = 6
    img_size = 128
    latent_dim = 32
    epochs = 50
    lr = 1e-3
    batch_size = 16
    k = 4
    return (
        batch_size,
        data_dir,
        epochs,
        feature_extractor,
        img_size,
        k,
        k_max,
        k_min,
        latent_dim,
        lr,
        out_dir,
        seed,
    )


@app.cell
def _(
    ConvAutoencoder,
    DataLoader,
    SAFImageDataset,
    batch_size,
    data_dir,
    epochs,
    extract_embeddings,
    extract_pretrained_embeddings,
    feature_extractor,
    glob,
    img_size,
    latent_dim,
    lr,
    np,
    os,
    out_dir,
    seed,
    torch,
    train_autoencoder,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    filepaths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No .png files found in {data_dir}")
    print(f"Found {len(filepaths)} images.")

    if feature_extractor == "autoencoder":
        train_ds = SAFImageDataset(filepaths, img_size=img_size, augment=True, grayscale=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = ConvAutoencoder(in_channels=3, latent_dim=latent_dim, img_size=img_size)
        model = train_autoencoder(model, train_loader, epochs=epochs, lr=lr, device=device)

        torch.save(model.state_dict(), os.path.join(out_dir, "autoencoder.pt"))

        embeddings, ordered_paths = extract_embeddings(model, filepaths, img_size, device, grayscale=False)
    else:
        embeddings, ordered_paths = extract_pretrained_embeddings(filepaths, img_size, device)
    return embeddings, ordered_paths


@app.cell
def _(
    choose_k_and_cluster,
    embeddings,
    k,
    k_max,
    k_min,
    ordered_paths,
    os,
    out_dir,
    parse_params,
    pd,
    seed,
):
    km, best_k, scores = choose_k_and_cluster(embeddings, k_min, k_max, k, seed)
    print(f"\nUsing k={best_k} clusters.")
    records = []
    for path, label in zip(ordered_paths, km.labels_):
        rec = {"filepath": path, "filename": os.path.basename(path), "cluster": int(label)}
        rec.update(parse_params(path))
        records.append(rec)
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "cluster_assignments.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote cluster assignments -> {csv_path}")
    return best_k, df, km


@app.cell
def _(df, os, out_dir):
    param_cols = ["D", "Ms", "T", "dmi", "Ku", "Sk"]
    summary = df.groupby("cluster")[param_cols].mean(numeric_only=True)
    print("\nMean physical parameters per cluster (for interpretation only):")
    print(summary)
    summary.to_csv(os.path.join(out_dir, "cluster_parameter_summary.csv"))
    return


@app.cell
def _(
    best_k,
    df,
    embeddings,
    km,
    os,
    out_dir,
    plot_cluster_grid,
    plot_embeddings_2d,
):
    plot_embeddings_2d(embeddings, km.labels_, os.path.join(out_dir, "embeddings_2d.png"))
    plot_cluster_grid(df, os.path.join(out_dir, f"cluster_grid_k{best_k}.png"))

    print(f"\nAll outputs written to: {os.path.abspath(out_dir)}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
