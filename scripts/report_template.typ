#let lr=sys.inputs.lr
#let epochs=sys.inputs.epochs
#let batch-size=sys.inputs.batch-size
#let best-epoch=sys.inputs.best-epoch
#let metrics-plot=image(sys.inputs.metrics-plot, width: 130%)
#let model-save-path=sys.inputs.model-save-path
#let dmi-value=sys.inputs.dmi-value
#let ku-value=sys.inputs.ku-value
#let pm-resolution=sys.inputs.pm-resolution
#let phase-diagram-img=image(sys.inputs.phase-diagram-img, width: 100%)

#title()[Reporte del entrenamiento]

= Parametros de entrada

- Learning rate: #lr
- Epochs: #epochs
- Batch Size: #batch-size

= Resultados del entrenamiento
- Best epoch at #best-epoch

#figure(
  metrics-plot
)

Modelo guardado en ruta `#model-save-path`

= Prediccion

El diagrama de fase predijo por el modelo a $J_"DMI"=#dmi-value$ y $K_u=#ku-value$ con una resolucion de res$=#pm-resolution$

#figure(
  phase-diagram-img
)