# Spintronics Neuromorphic

## What is neuromorphic computation
The idea is to emulate the energy-efficient hardware the is the brain to perform highly sophisticated computational tasks. A core characteristic of the brain is that it doesnt
separate the core memory and processing units which would slow it down and increase the energy consumption. Another important fact is that the brain performs low-precision calculations
as opposed to modern super computers. So the idea is to build artifical neurons and synapses, connect them together in huge numbers, organize them in complex systems and compute
with them efficiently.
```{figure} images/vs_arch_comp.png
:label: arch-comp-diff
:alt: Differences in computing architectures
:align: center
:width: 550 px

Differences in computing architectures [^neuromorphic_thesis]
```
## Spintronics applications on neuromorphic computation
However using CMOS technology to emulate neurons is way too energy consuming.
Thats why we explore other physical systems to build this neuromorphic systems, example of these are spintronics nanodevices using skyrmions are particularly interesting because:
- They are extremely small and behave like particles that can be created and annihilated with a low energy cost.
- Multiple nanoscale skyrmions can accumulate within a defined device area without interacting with topographic defects.
There are a couple of demonstrations based on single devices emulations however what is lacking is the system level emulation. A crucial factor limiting this system level neuromorphic
computing is the restriction of simulation tools. In spintronics, the micromagnetic model is a fundamental framework to describe theoretically the magnetization process on the micron
scale. Some study of physical phenomena like Spin Transfer Torque (STT), Spin-Hall and Spin-Seebeck effects allow for the creation of new devices:

- Spin Torque Oscillators
- Spin-transfer torque magnetoresistive random acces memory
- skyrmions

However there are some weaknesses to micromagnetic simulations like

- Time-consuming simulations for large systems ($\approx 1\mu m$) or longer time scales.
- In a large parameter space, the simulation needs to re execute from scratch. or longer time scales.
- Re-execution of the simulation for a large parameter space..
## Relation to Reservoir Computing




# Questions

1. What is CMOS tech?
2. What topographic defects are there?

[^neuromorphic_thesis]: [Thesis]("https://theses.hal.science/tel-03770225") about Modeling and simulations of skyrmionic neuromorphic applications
