#let lr=sys.inputs.lr
#let epochs=sys.inputs.epochs
#let batch-size=sys.inputs.batch-size
#let dmi-value=sys.inputs.dmi-value
#let ku-value=sys.inputs.ku-value
#let pm-resolution=sys.inputs.pm-resolution

#title()[Reporte del entrenamiento]

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

El diagrama de fase predicho por el modelo a $J_"DMI"=#dmi-value$ y $K_u=#ku-value$ con una resolucion de res$=#pm-resolution$

#figure(
  image(sys.inputs.at("dnn_do-phase-diagram-img"), width: 120%)
)

== Comparacion con data no entrenada

#figure(
  image(sys.inputs.at("dnn_do-oor-img"), width: 100%)
)

= Resultados del entrenamiento modelo BatchNorm
- Best epoch at #sys.inputs.at("dnn_batch-best-epoch")

#figure(
  image(sys.inputs.at("dnn_batch-metrics-plot"), width: 100%)
)

Modelo guardado en ruta #sys.inputs.at("dnn_batch-model-save-path")

= Diagrama de Fase Predicho

El diagrama de fase predicho por el modelo a $J_"DMI"=#dmi-value$ y $K_u=#ku-value$ con una resolucion de res$=#pm-resolution$

#figure(
  image(sys.inputs.at("dnn_batch-phase-diagram-img"), width: 100%)
)

= Comparacion con data no entrenada

#figure(
  image(sys.inputs.at("dnn_batch-oor-img"), width: 120%)
)