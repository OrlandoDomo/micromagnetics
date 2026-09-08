import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np

    return (pl,)


@app.cell
def _(pl):
    csv_file = '../data/csv_data/saf_relax-hi_res.csv'
    df = pl.read_csv(csv_file).with_columns([
        pl.when(abs(pl.col("S2k_bot") - 1) < 0.25)
          .then(1)
          .otherwise(0)
          .alias("phase_label")
      ])
    df.write_csv('../data/csv_data/saf_relax-hi_res-labeled.csv')

    df2 = df.with_columns(
        pl.when((pl.col('S2k_bot') > 0.25) & (pl.col('S2k_bot') < 0.75)).then(2)
        .when((pl.col('S2k_bot') >= 0) & (pl.col('S2k_bot') <= 0.25)).then(0)
        .when((pl.col('S2k_bot') >= 0.75) & (pl.col('S2k_bot') <= 1.25)).then(1)
        .when((pl.col('S2k_bot') > 1.25) & (pl.col('S2k_bot') <= 1.9)).then(3)
        .when((pl.col('S2k_bot') > 1.9) & (pl.col('S2k_bot') <= 2.1)).then(4)
        .when(pl.col('S2k_bot') > 2.1).then(5)    
        .otherwise(6)
        .alias('phase_label')
    )
    df2.write_csv('../data/csv_data/saf_relax-hi_res-multi-labeled.csv')
    return (df2,)


@app.cell
def _(df2):
    df2
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
