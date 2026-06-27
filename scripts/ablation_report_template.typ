// ==========================================
// DOCUMENT SETUP
// ==========================================
#set page("a4", margin: (x: 1.5cm, y: 2cm))
//#set text(font: "Linux Libertine", size: 10pt)
#set heading(numbering: "1.1.")

// ==========================================
// TITLE & GLOBAL CONFIGURATION
// ==========================================
#align(center)[
  #text(18pt, weight: "bold")[Ablation Study: Neural Network Phase Diagram Prediction]
  #v(1em)
  #text(12pt, style: "italic")[Automated Model Comparison & Out-of-Distribution Evaluation]
]
#v(2em)

= Global Run Configuration
#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  fill: (col, row) => if row == 0 { luma(230) } else { white },
  align: center,
  [*Max Epochs*], [*Batch Size*], [*Learning Rate*], [*Resolution*],
  sys.inputs.at("epochs", default: "N/A"),
  sys.inputs.at("batch-size", default: "N/A"),
  sys.inputs.at("lr", default: "N/A"),
  sys.inputs.at("pm-resolution", default: "N/A")
)

#v(1em)
#line(length: 100%, stroke: 0.5pt + luma(150))
#v(1em)

// ==========================================
// RENDER COMPONENT (Reusable Block)
// ==========================================
// This function dynamically pulls the images and text for a specific 
// model and condition combination so we don't repeat code 6 times.
#let render-condition(model_name, cond_name) = {
  let prefix = model_name + "-" + cond_name

  heading(level: 3)[Condition: #cond_name]
  
  text(weight: "semibold")[Achieved Best Validation Loss at Epoch: ]
  sys.inputs.at(prefix + "-best-epoch", default: "N/A")
  v(0.5em)

  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 2%,
    figure(
      image(sys.inputs.at(prefix + "-metrics-plot", default: ""), width: 100%),
      caption: [Training Loss & F1]
    ),
    figure(
      image(sys.inputs.at(prefix + "-phase-map-img", default: ""), width: 100%),
      caption: [Phase Map (DMI=#sys.inputs.at("dmi-value"), Ku=#sys.inputs.at("ku-value"))]
    ),
    figure(
      image(sys.inputs.at(prefix + "-oor-img", default: ""), width: 100%),
      caption: [OOD Interpolation Matrix]
    )
  )
  v(2em)
}
// ==========================================
// RESULTS: ARCHITECTURE 1 (DROPOUT)
// ==========================================
= Architecture: DenseNetwork (Dropout)
This section evaluates the Dropout-based architecture across the three dataset conditions to observe variance in predictive stability.

#render-condition("dnn_do", "Baseline")
#render-condition("dnn_do", "Augmented")
#render-condition("dnn_do", "Engineered")

#pagebreak() // Push the second model to a fresh page

// ==========================================
// RESULTS: ARCHITECTURE 2 (BATCHNORM)
// ==========================================
= Architecture: DenseNetwork (BatchNorm)
This section evaluates the Batch Normalization-based architecture across the three dataset conditions to observe variance in predictive stability.

#render-condition("dnn_batch", "Baseline")
#render-condition("dnn_batch", "Augmented")
#render-condition("dnn_batch", "Engineered")