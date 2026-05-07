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

    return BytesIO, Image, alt, base64, os, pl


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
def _():
    return


if __name__ == "__main__":
    app.run()
