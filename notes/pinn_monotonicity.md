# Monotonicity in PINNs

I came across this concept of monotonicity in Physics-Informed Neural Networks (PINNs) that talk about adding a way to penalize non monotonic changes 
to better represent the hidden physical laws that govern a system. 

For example <doi:10.1016/j.fuel.2024.131026> use a monotonic relations to improve the predictions made by the PINNs for $\text{NO}_x$ emissions. [](https://doi.org/10.48550/arXiv.2511.14348) use monotonicity to enforce the second law of thermodynamics is obeyed by the solutions of PINNs of different known PDEs in physics.
Another one used for fluid dynamics [](https://doi.org/10.1063/5.0334590)

So it occurred to me that i can use this to enforce my ML model to obey some kind of monotonic relationship of the topological charge against some magnetic parameter such as $M_s$, $J_{\text{DMI}}$, $K_u$, etc. However, I thought this monotnic relation would not necessarily hold in phase transitions boundaries such as skyrmion -> skyrmionium. 

So far, it seems there really isnt any study about any monotonic relationship of the topological charge $S_k$ with respect to any magnetic parameter used in simulations; also, how would one include the phase transition effect, cuz in transition boundaries $S_k$ could change in the other direction, meaning the ML model would penalize this boundary and not predict correctly. 

So I searched some papers related to phase transitions and monotonicity, not necesarilly talking about $S_k$ tho. [](https://doi.org/10.1007/BF01029985) and [](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.180601) which just classifies phase transition orders according to some inflection points, meaning change of the monotonicity. Dont know if they would be really useful, need more read. However, since this talks about the energy, maybe i can plot some $S_k$ vs Energy plots to see if i can observe some sharp inflections corresponding to the phase transitions, idk. So i searched if someone already did that and no results so far. However, when searching about energy regimes, i came across this new paper [](https://www.nature.com/articles/s41563-026-02673-9) which uses the method Return Point Memory to probe the thermodynamic stability and nucleation channels for their skyrmions which could be really useful for my work.