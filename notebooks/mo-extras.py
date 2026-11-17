import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np

    return (pl,)


@app.cell
def _(pl):
    def export_labeled_data():
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

    return


@app.cell
def _(pl):
    def export_extended_data():
        from pathlib import Path
        import discretisedfield as df
        import discretisedfield.tools as dft

        rows = []
        ovf_files = Path("..\ovf_files\saf_results_relax_extended")
        d = 450 # nm
        Ms = 300 # kA/m
        Ku = 0.1 # MJ/m^3
        dmis = [x/100 for x in range(50,105,5)]
        rkkys = [0.0,0.25,0.5,1.0,1.5,2.0,4.0,8.0]

        for dmi in dmis:
            for rkky in rkkys:
                if rkky == 0.25:
                    ovf_file = ovf_files / f'dmi={dmi}' / f'm_D={d}_Ms={Ms}_T=0_dmi={dmi}_Ku={Ku}_RKKY=-{rkky:.2f}.ovf'
                else:
                    ovf_file = ovf_files / f'dmi={dmi}' / f'm_D={d}_Ms={Ms}_T=0_dmi={dmi}_Ku={Ku}_RKKY=-{rkky:.1f}.ovf'

                print(rkky)
                if ovf_file.exists():
                    read_field = df.Field.from_file(ovf_file)
                    s2k_bot = dft.topological_charge(read_field.sel(z=0.5e-9), absolute=True)
                    s2k_top = dft.topological_charge(read_field.sel(z=2.5e-9), absolute=True)
                
                    rows.append([d, Ms, dmi, Ku, rkky*-0.2, s2k_bot, s2k_top])
                
        df = pl.DataFrame(rows, schema=['D', 'Ms', 'DMI', 'Ku', 'RKKY', 's2k_bot', 's2k_top'], orient="row").with_columns(
            pl.when((pl.col('s2k_bot') > 0.25) & (pl.col('s2k_bot') < 0.75)).then(2) # Exotico
            .when((pl.col('s2k_bot') >= 0) & (pl.col('s2k_bot') <= 0.25)).then(0) # FM
            .when((pl.col('s2k_bot') >= 0.75) & (pl.col('s2k_bot') <= 1.25)).then(1) # Skyrmion
            .when((pl.col('s2k_bot') > 1.25) & (pl.col('s2k_bot') <= 1.8)).then(3) # Complejo
            .when((pl.col('s2k_bot') > 1.8) & (pl.col('s2k_bot') <= 2.15)).then(4) # Skyrmionium
            .when(pl.col('s2k_bot') > 2.15).then(5) # Laberinto
            .otherwise(6) # extras
            .alias('phase_label')
        )
        print(df)
        df.write_csv('..\data\csv_data\saf-relax-extended.csv')
    
        return 1

    return (export_extended_data,)


@app.cell
def _(export_extended_data):
    export_extended_data()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
