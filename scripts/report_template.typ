#let lr=sys.inputs.lr
#let epochs=sys.inputs.epochs
#let batch-size=sys.inputs.batch-size
#let best-epoch=sys.inputs.best-epoch
#let metrics-plot=image(sys.inputs.metrics-plot, width: 130%)

#title()[Reporte del entrenamiento]

= Parametros de entrada

- Learning rate: #lr
- Epochs: #epochs
- Batch Size: #batch-size

= Resultados
- Best epoch at #best-epoch

#figure(
  metrics-plot
)
asdasd