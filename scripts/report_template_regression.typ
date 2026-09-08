#let lr=sys.inputs.lr
#let epochs=sys.inputs.epochs
#let batch-size=sys.inputs.batch-size
#let dmi-value=sys.inputs.dmi-value
#let ku-value=sys.inputs.ku-value

#title()[Reporte del entrenamiento de regresion]

= Parametros de entrada

- Learning rate: #lr
- Epochs: #epochs
- Batch Size: #batch-size

= Resultados del entrenamiento modelo DropOut
- Best epoch at #sys.inputs.at("dnn_do-best-epoch")

#figure(
  image(sys.inputs.at("dnn_do-metrics-plot"), width: 100%)
)

Modelo guardado en ruta #sys.inputs.at("dnn_do-model-save-path")

== Diagrama de Fase Predicho

El diagrama de fase predicho por el modelo a $J_"DMI"=#dmi-value$ y $K_u=#ku-value$

#figure(
  image(sys.inputs.at("dnn_do-phase-diagram-img"), width: 120%)
)

#figure(
  image(sys.inputs.at("dnn_do-metrics-img-predicted"), width: 120%)
)

== Comparado con new data

#figure(
  image(sys.inputs.at("dnn_do-phase-diagram-img-unseen"), width: 120%)
)

#figure(
  image(sys.inputs.at("dnn_do-metrics-img-unseen"), width: 120%)
)

= Resultados del entrenamiento modelo BatchNorm
- Best epoch at #sys.inputs.at("dnn_batch-best-epoch")

#figure(
  image(sys.inputs.at("dnn_batch-metrics-plot"), width: 100%)
)

Modelo guardado en ruta #sys.inputs.at("dnn_batch-model-save-path")

= Diagrama de Fase Predicho

El diagrama de fase predicho por el modelo a $J_"DMI"=#dmi-value$ y $K_u=#ku-value$

#figure(
  image(sys.inputs.at("dnn_batch-phase-diagram-img"), width: 100%)
)

#figure(
  image(sys.inputs.at("dnn_batch-metrics-img-predicted"), width: 100%)
)

== Comparado con new data

#figure(
  image(sys.inputs.at("dnn_batch-phase-diagram-img-unseen"), width: 100%)
)

#figure(
  image(sys.inputs.at("dnn_batch-metrics-img-unseen"), width: 100%)
)
