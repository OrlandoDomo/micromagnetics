import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    return duckdb, pl


@app.cell
def _(pl):
    df = pl.read_csv("..\data\csv_data\saf_relax-results.csv").drop([""])
    new_df = pl.read_csv("..\data\csv_data\saf_relax-dmi=0.8_ku=0.08.csv").drop([""])
    return


@app.cell
def _(duckdb):
    con = duckdb.connect("../data/duckdb_saf.db")
    return (con,)


@app.cell
def _(con):
    #con.sql("CREATE TABLE IF NOT EXISTS saf_results AS SELECT * FROM df")
    #con.sql("INSERT INTO saf_results SELECT * FROM new_df")
    con.table("saf_results").show()
    return


@app.cell
def _(con):
    con.sql("SELECT * FROM saf_results WHERE DMI=0.8")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
