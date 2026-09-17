import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import re
    import glob
    import struct
    import warnings
 
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
 
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    return (
        DataLoader,
        Dataset,
        F,
        KMeans,
        PCA,
        glob,
        nn,
        np,
        os,
        pd,
        plt,
        re,
        silhouette_score,
        struct,
        torch,
        warnings,
    )


@app.cell
def _(np, os, re, warnings):
    PARAM_PATTERN = re.compile(
        r"D=(?P<D>-?[\d.]+)_Ms=(?P<Ms>-?[\d.]+)_T=(?P<T>-?[\d.]+)_"
        r"dmi=(?P<dmi>-?[\d.]+)_Ku=(?P<Ku>-?[\d.]+)"
    )
 
 
    def parse_params(filename: str) -> dict:
        match = PARAM_PATTERN.search(os.path.basename(filename.split('.ovf')[0]))
        if match is None:
            warnings.warn(f"Could not parse parameters from filename: {filename}")
            return {k: np.nan for k in ["D", "Ms", "T", "dmi", "Ku", "Sk"]}
        return {k: float(v) for k, v in match.groupdict().items()}

    return (parse_params,)


@app.cell
def _(np, struct, warnings):
    def parse_ovf(filepath, z_index=0):
        """Read an .ovf file and return a (H, W, valuedim) float32 numpy array
        for the requested z-slice, plus a dict of header metadata.
 
        Handles the standard OOMMF/mumax3 OVF layout: a text header of
        "# key: value" lines, then a "# Begin: Data <fmt>" line followed by the
        data block, then "# End: Data <fmt>".
        """
        with open(filepath, "rb") as f:
            raw = f.read()
 
        # Header is always plain ASCII text up to "# Begin: Data"
        header_end = raw.find(b"# Begin: Data")
        if header_end == -1:
            raise ValueError(f"Could not find data block in {filepath}")
        header_text = raw[:header_end].decode("ascii", errors="ignore")
 
        meta = {}
        for line in header_text.splitlines():
            line = line.strip()
            if not line.startswith("#"):
                continue
            line = line.lstrip("#").strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
 
        def get_int(key, default=1):
            return int(float(meta.get(key, default)))
 
        xnodes = get_int("xnodes", 1)
        ynodes = get_int("ynodes", 1)
        znodes = get_int("znodes", 1)
        valuedim = get_int("valuedim", 3)
 
        # Find the format line, e.g. "# Begin: Data Binary 4" / "Text" / "Binary 8"
        fmt_line_end = raw.find(b"\n", header_end)
        fmt_line = raw[header_end:fmt_line_end].decode("ascii", errors="ignore")
        fmt = fmt_line.split("Data", 1)[1].strip().lower()
 
        data_start = fmt_line_end + 1
        n_values = xnodes * ynodes * znodes * valuedim
 
        if fmt.startswith("text"):
            end_marker = raw.find(b"# End: Data", data_start)
            block_text = raw[data_start:end_marker].decode("ascii", errors="ignore")
            values = np.fromstring(block_text, sep=" ", dtype=np.float64)
            values = values[:n_values]
        elif fmt.startswith("binary 4"):
            # 4-byte float check value = 1234567.0
            check = struct.unpack_from("<f", raw, data_start)[0]
            offset = data_start + 4
            if abs(check - 1234567.0) > 1.0:
                # try big-endian
                check_be = struct.unpack_from(">f", raw, data_start)[0]
                endian = ">" if abs(check_be - 1234567.0) < 1.0 else "<"
            else:
                endian = "<"
            values = np.frombuffer(raw, dtype=f"{endian}f4", count=n_values, offset=offset).astype(np.float64)
        elif fmt.startswith("binary 8"):
            check = struct.unpack_from("<d", raw, data_start)[0]
            offset = data_start + 8
            if abs(check - 123456789012345.0) > 1.0:
                check_be = struct.unpack_from(">d", raw, data_start)[0]
                endian = ">" if abs(check_be - 123456789012345.0) < 1.0 else "<"
            else:
                endian = "<"
            values = np.frombuffer(raw, dtype=f"{endian}f8", count=n_values, offset=offset).astype(np.float64)
        else:
            raise ValueError(f"Unsupported OVF data format '{fmt}' in {filepath}")
 
        # Data is ordered x fastest, then y, then z, with valuedim components
        # interleaved per node: (z, y, x, valuedim)
        arr = values.reshape(znodes, ynodes, xnodes, valuedim)
 
        if z_index >= znodes:
            warnings.warn(f"{filepath}: requested z_index={z_index} but znodes={znodes}; using 0.")
            z_index = 0
        arr2d = arr[z_index]  # (ynodes, xnodes, valuedim)
 
        return arr2d.astype(np.float32), meta

    return (parse_ovf,)


@app.cell
def _(Dataset, F, np, parse_ovf, torch):
    class SAFOVFDataset(Dataset):
        def __init__(self, filepaths, img_size=128, augment=False, z_index=0, normalize_unit=True):
            self.filepaths = filepaths
            self.img_size = img_size
            self.augment = augment
            self.z_index = z_index
            self.normalize_unit = normalize_unit
 
        def __len__(self):
            return len(self.filepaths)
 
        def _load(self, path):
            arr, _ = parse_ovf(path, z_index=self.z_index)  # (H, W, 3) = (Mx, My, Mz)
            if self.normalize_unit:
                mag = np.linalg.norm(arr, axis=-1, keepdims=True)
                mag = np.clip(mag, 1e-12, None)
                arr = arr / mag
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()  # (3, H, W)
            tensor = F.interpolate(tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                                    mode="bilinear", align_corners=False).squeeze(0)
            return tensor
 
        def __getitem__(self, idx):
            path = self.filepaths[idx]
            tensor = self._load(path)
 
            if self.augment:
                if torch.rand(1).item() < 0.5:
                    tensor = torch.flip(tensor, dims=[2])  # flip width (horizontal)
                    tensor[0] = -tensor[0]  # Mx sign flips under left-right mirror
                if torch.rand(1).item() < 0.5:
                    tensor = torch.flip(tensor, dims=[1])  # flip height (vertical)
                    tensor[1] = -tensor[1]  # My sign flips under up-down mirror
 
            return tensor, path

    return (SAFOVFDataset,)


@app.cell
def _(nn):
    class ConvAutoencoder(nn.Module):
        def __init__(self, in_channels=3, latent_dim=32, img_size=128):
            super().__init__()
            self.img_size = img_size
 
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
                nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), nn.Tanh(),
                # Tanh (not Sigmoid) because targets are unit-vector components in [-1, 1]
            )
 
        def encode(self, x):
            h = self.encoder_conv(x)
            h = h.flatten(1)
            return self.to_latent(h)
 
        def decode(self, z):
            h = self.from_latent(z)
            h = h.view(-1, 64, self.reduced_size, self.reduced_size)
            return self.decoder_conv(h)
 
        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

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
            for tensors, _ in loader:
                tensors = tensors.to(device)
                optimizer.zero_grad()
                recon, _ = model(tensors)
                loss = criterion(recon, tensors)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * tensors.size(0)
            avg_loss = total_loss / len(loader.dataset)
            if epoch == 1 or epoch % max(1, epochs // 20) == 0 or epoch == epochs:
                print(f"Epoch {epoch:4d}/{epochs}  |  reconstruction MSE: {avg_loss:.5f}")
        return model

    return (train_autoencoder,)


@app.cell
def _(DataLoader, SAFOVFDataset, np, torch):
    @torch.no_grad()
    def extract_embeddings(model, filepaths, img_size, device, z_index=0):
        model.eval()
        dataset = SAFOVFDataset(filepaths, img_size=img_size, augment=False, z_index=z_index)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
 
        embeddings, ordered_paths = [], []
        for tensors, paths in loader:
            tensors = tensors.to(device)
            z = model.encode(tensors)
            embeddings.append(z.cpu().numpy())
            ordered_paths.extend(paths)
        return np.concatenate(embeddings, axis=0), ordered_paths

    return (extract_embeddings,)


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
def _(PCA, np, parse_ovf, plt):
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

    def plot_embeddings_by_param(embeddings, df, out_path,
                                  param_cols=("D", "Ms", "T", "dmi", "Ku")):
        """Same PCA projection as plot_embeddings_2d, but one subplot per physical
        parameter, colored continuously, to see which parameter(s) actually drive
        the layout (e.g. the arc/curve shape or the main axis split)."""
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
 
        ncols = 3
        nrows = int(np.ceil(len(param_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
 
        for i, col in enumerate(param_cols):
            ax = axes[i]
            vals = df[col].values
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="viridis",
                             s=55, edgecolor="k", linewidth=0.3)
            fig.colorbar(sc, ax=ax, label=col)
            ax.set_title(f"Colored by {col}")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
 
        for j in range(len(param_cols), len(axes)):
            axes[j].axis("off")
 
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
 
    def plot_cluster_grid(df, out_path, z_index=0, n_per_cluster=6):
        """Render the Mz component with a red/white/blue colormap, matching the
        look of the original PNGs, for a few example files per cluster."""
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
                    arr, _ = parse_ovf(path, z_index=z_index)
                    mz = arr[..., 2]
                    vmax = np.abs(mz).max() or 1.0
                    ax.imshow(mz, cmap="bwr", vmin=-vmax, vmax=vmax)
                    if col == 0:
                        ax.set_title(f"cluster {c}", loc="left", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    return plot_cluster_grid, plot_embeddings_2d, plot_embeddings_by_param


@app.cell
def _():
    seed = 123
    out_dir = '../data/unsupervised-test'
    data_dir = '../ovf_files/saf_results/dmi=0.8'
    z_index = 0
    k_min = 4
    k_max = 6
    img_size = 128
    latent_dim = 32
    epochs = 200
    lr = 1e-3
    batch_size = 16
    k = None
    no_unit_normalize = True
    return (
        batch_size,
        data_dir,
        epochs,
        img_size,
        k,
        k_max,
        k_min,
        latent_dim,
        lr,
        no_unit_normalize,
        out_dir,
        seed,
        z_index,
    )


@app.cell
def _(data_dir, glob, np, os, out_dir, seed, torch):
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    filepaths = sorted(glob.glob(os.path.join(data_dir, "*.ovf")))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No .ovf files found in {data_dir}")
    print(f"Found {len(filepaths)} .ovf files.")
    return device, filepaths


@app.cell
def _(
    ConvAutoencoder,
    DataLoader,
    SAFOVFDataset,
    batch_size,
    device,
    epochs,
    extract_embeddings,
    filepaths,
    img_size,
    latent_dim,
    lr,
    no_unit_normalize,
    os,
    out_dir,
    torch,
    train_autoencoder,
    z_index,
):
    train_ds = SAFOVFDataset(filepaths, img_size=img_size, augment=True,
                              z_index=z_index, normalize_unit=not no_unit_normalize)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = ConvAutoencoder(in_channels=3, latent_dim=latent_dim, img_size=img_size)
    model = train_autoencoder(model, train_loader, epochs=epochs, lr=lr, device=device)
    torch.save(model.state_dict(), os.path.join(out_dir, "autoencoder.pt"))

    embeddings, ordered_paths = extract_embeddings(model, filepaths, img_size, device, z_index=z_index)
    return embeddings, ordered_paths


@app.cell
def _(choose_k_and_cluster, embeddings, k, k_max, k_min, seed):
    km, best_k, scores = choose_k_and_cluster(embeddings, k_min, k_max, k, seed)
    print(f"\nUsing k={best_k} clusters.")
    return best_k, km


@app.cell
def _(
    best_k,
    embeddings,
    km,
    ordered_paths,
    os,
    out_dir,
    parse_params,
    pd,
    plot_cluster_grid,
    plot_embeddings_2d,
    plot_embeddings_by_param,
    z_index,
):
    records = []
    for path, label in zip(ordered_paths, km.labels_):
        rec = {"filepath": path, "filename": os.path.basename(path), "cluster": int(label)}
        rec.update(parse_params(path))
        records.append(rec)
    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "cluster_assignments.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote cluster assignments -> {csv_path}")

    param_cols = ["D", "Ms", "T", "dmi", "Ku"]
    summary = df.groupby("cluster")[param_cols].mean(numeric_only=True)
    print("\nMean physical parameters per cluster (for interpretation only):")
    print(summary)
    summary.to_csv(os.path.join(out_dir, "cluster_parameter_summary.csv"))

    plot_embeddings_2d(embeddings, km.labels_, os.path.join(out_dir, "embeddings_2d.png"))
    plot_embeddings_by_param(embeddings, df, os.path.join(out_dir, "embeddings_2d_by_param.png"))
    plot_cluster_grid(df, os.path.join(out_dir, f"cluster_grid_k{best_k}.png"), z_index=z_index)

    print(f"\nAll outputs written to: {os.path.abspath(out_dir)}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
