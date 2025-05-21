# Micromagnetics Fundamentals
First, micromagnetics is the usage of continuum models to study patterns of magnetization in magnetic materials at the scale of nanometers. In this sense, micromagnetics uses computation to determine the spatial distribution of magnetization in magnetic materials.

## Equations of magnetodynamics
Magnetization is a vector field $\mathbf{M}$, its dynamics is described by the Landau-Lifshitz-Gilbert equation which links the magnetizacion direction at each point to a magnetic field $\mathbf{H}$ and some paremeters dependent on the material.
```{math}
:label: llg-eq
\frac{d\mathbf{M}}{dt} = - |\gamma| \mathbf{M} \times \mathbf{H} 
+ \frac{\alpha}{|\mathbf{M}|} \left( \mathbf{M} \times \frac{d\mathbf{M}}{dt} \right)
```
Both quantities $\mathbf{M}$ and $\mathbf{H}$ are measured in $[A/m]$, physically:

- The first term on the right, $- |\gamma| \mathbf{M} \times \mathbf{H}$, describes
the precession of the magnetization about the magnetic field and the quantity 
$ \gamma$ is called the gyromagnetic ratio typically given the value 
$|\gamma|=2.21 \times 10^5 \frac{m}{As}$ which in turn is the frequency of precession
of the free electron spin in the presence of the same magnetic field.
- The second term: $\frac{\alpha}{|\mathbf{M}|} \left( \mathbf{M} \times \frac{d\mathbf{M}}{dt} \right)$ describes the loss of energy due to friction or damping against the rotation of the magnetization.

### Energy
The total energy of a system $W$ is the just the integral of the energy density $E$
```{math}
:label: energy-eq
W = \int_V E(r) d^3 r
```
- **Energy density $E$** is a function of the magnetization $\mathbf{M}$ and the position $\mathbf{r}$. 
- **Total energy** $W$ is a functional of the magnetization $W=W[\mathbf{M}]$
- **Effective magnetic field** $H$ is the variational derivative $\mathbf{H}=-\frac{1}{\mu_0} \frac{\delta W}{\delta \mathbf{M}}$

Historically, Landau-Lifhitz proposed the first eqation for magnetodynamics as
```{math}
:label: ll-eq
\frac{d\mathbf{M}}{dt} = - \bar{\gamma} \mathbf{M} \times \mathbf{H} 
+ \frac{\lambda}{|\mathbf{M}|} \mathbf{M} \times \mathbf{H} \times \mathbf{M}
```
There is a relation between both constants in both equations:
```{math}
\begin{align*}
-\bar{\gamma} &= \frac{|\gamma|}{1+\alpha^2} \\
\lambda &= \frac{|\gamma|\alpha}{1+\alpha^2}
\end{align*}
```
we can notice some things when looking at both forms:

1. The LL formulation [](#ll-eq) is easier to work with numerically since there is 
no derivative term $d\mathbf{M}/dt$.
2. In the LLG formulation [](#llg-eq), if we increase $\alpha$ infinitelly then we 
could have infinite damping, but if we transform back to hte LL formulation and see
the relations between the constants, both $\lambda$ and $\bar{\gamma}$ just shrink. 
3. In both forms is clear that the magnetization is orthogonal to its temporal change,
meaning there is only change in direction and the magnitude remains constant.
4. According to the equation for the magnetic field [](#energy-eq) we see that $\mathbf{H}$ is aligned with the lower energy direction in
the surface over the space of magnetization $W$, therefore motion perpendicular to $\mathbf{H}$ is energy neutral
5. The damping term in the LL equation []($ll-eq) tends to align $\mathbf{M}$ with $\mathbf{H}$ lowering the energy (?).
6. When the magnetization and the magnetic field are paralallel then we a stationary configuration $d\mathbf{M}/dt=0$ and from [](#energy-eq) this corresponds to a local energy minima.

Since terminating configurations of the LLG equation are local minima of the energy, in most cases we just want direct energy minimization methods to determine the energy-minimizing
magnetization states. The technique known as quasi-static micromagnetics computes the local energy minimun at one applied field held static, then after
the equilibrium configuration is found the applied field is changed slightly and then a new energy minimizing configuration is sought.

## Magnetic eneryg components

There are four fundamental sources in micromagnetics:

1. Anisotropy Energy: $\mathbf{H(r)}$ depend on $\mathbf{M}$ only on the point $r$ itself
2. Quantum mechanical Exchange Energy: $\mathbf{H(r)}$ depend on $\mathbf{M}$ in a small neighbordhood of $r$
3. Self-magnetostatic (dipole-dipole): $\mathbf{H(r)}$ depends on $\mathbf{M}$ globally across the entire volume
4. Zeeman Energy: $\mathbf{H(r)}$ is independent of $\mathbf{M}$ because it represents the influence of the environment outside of the material of interest.

### Anisotropy Energy

Electron spins interacting with the atomic structure is such that magnetization in certain directions is favored over others. For example, a materail with single favored
direciton along an axis determined by the unit vector $\mathbf{u}$ may be represented by the energy term
```{math}
W_K = \int_V -K \left( \frac{\mathbf{M}}{|\mathbf{M}|} \cdot \mathbf{u} \right)^2 d^3r
```
That material is said to have a uniaxial anisotropy. This is characterized by the energy density $K [J/m^3]$, if $K>0$ then the direction $\mathbf{u}$ is energetically preferred for
magnetization; and if $K<0$ then it is hard axis, meaning is favorable for the magnetization to lie in the plane orthogonal to this direction (easy plane for magnetization).
In this case, according to [](#energy-eq) the anisotropy magnetic field is left as 
```{math}
\mathbf{H}_K = \frac{2K}{\mu_0 |\mathbf{M}|^2} (\mathbf{M} \cdot \mathbf{u})\mathbf{u}
```
When the anisotropy favors multiple axes, analogous energy terms can be specified; and in many cases the crystalline structure of the material is reflected in this energy term, which
is known as the magneto-crystalline anisotropy

### Exchange Energy

Physically, the magnetic moments of electrons the exchange interaction provokes the alignment of neighboring electrons' spin (spin coupling) resulting in the formation of magnetic
domains within magnetic materials. In the scope of micromagnetics we can represent this exchange interaction by penalizing large spatial rates of change in magnetization, meaning 
the gradient of the magnetizaion should vary slightly and continously across magnetic domains. A convetional formulation for exchange energy is given by
```{math}
W_{\text{exch}} = \int_V \frac{A}{|\mathbf{M}|^2} (|\nabla M_x|^2 + |nabla M_y|^2 + |\nabla M_z|^2) d^3r
```
where $A$ is an exchange coefficiente measured in [J/m]. We can rewrite the above and calculate the corresponding exchange field $\mathbf{H}$:
```{math}
\begin{align*}
W_{\text{exch}} &= \int_V - \frac{A}{|\mathbf{M}|^2} \mathbf{M} \cdot \left(\frac{\partial\mathbf{M}}{\partial x^2} 
    + \frac{\partial\mathbf{M}}{\partial y^2}
    + \frac{\partial\mathbf{M}}{\partial x^2}\right) d^3r \\

\mathbf{H}_{\text{exch}} &= \frac{2A}{\mu_0|\mathbf{M}|^2} \left(\frac{\partial\mathbf{M}}{\partial x^2} 
    + \frac{\partial\mathbf{M}}{\partial y^2}
    + \frac{\partial\mathbf{M}}{\partial x^2}\right) 
\end{align*}
```
### Self-magnetostatic energy

The magnetization gives rise naturally to a demagnetizing field which diminishes the effect of an external field on the magnetic field inside the magnet, as if it "prevents it
from entering" [^demag-field]. If we imagine our continuum representation of magnetization as an approximation to a collection of discrete elementary magnets, the magnetostatic field
is the field to represent the sum of the dipole-dipole interactions. For a volume $V$ bound by a closed surface $S$ with normal vector $\mathbf{n}$, the magnetostatic field is computed as

```{math}
\mathbf{H}_{\text{demag}}(\mathbf{r}) = \frac{1}{4\pi} \int_V -\nabla'\cdot \mathbf{M}(\mathbf{r}') \frac{\mathbf{r}-\mathbf{r}'}{|\mathbf{r}-\mathbf{r}'|^3} d^3\mathbf{r}'
+ \frac{1}{4\pi} \int_S \mathbf{n}\cdot \mathbf{M}(\mathbf{r}') \frac{\mathbf{r}-\mathbf{r}'}{|\mathbf{r}-\mathbf{r}'|^3} d^3\mathbf{r}'
```

[^demag-field]: https://www.phys.ksu.edu/personal/wysin/notes/demag.pdf

We've seen that exchange energy favors the formation of magnetic domains and now the magnetostatic energy tends to favor anti-parallel alignments that break up these domains.
In this sense, the task of micromagnetics is often the computation of what equilibrium arises from these competing tendencies. We can do this looking at the ratio between the 
quantities: exchange coefficient $A$ which characterizes the exchange energy and the saturation magnetization $M_s=|\mathbf{M}|$ characterizing the magnetostatic energy in the 
expresion for the quantity known as the magnetostatic exchange length in meters:
```{math}
l_{\text{ex}} = \sqrt{\frac{2A}{\mu_0 M_s^2}} [m]
```
This is tipically around 5 nm for the more common magnetic materials.

- For *magnetically soft materials*, the ones with low anisotropy this is the scale of spatial features in energy minimizing magnetization configurations, providing a reference for
the required spatial resolution of discretizations in micromagnetic simulations.
- For *magnetically hard materials* another distance of interest is the magnetocrystalline-exchange length, given by:
```{math}
l_{\text{ex},K} = \sqrt{\frac{A}{K}} [m]
```

### Zeeman Energy

This is just the effect of an outside magnetic field applied to the volumen of interest $\mathbf{H}_{app}$, the relation then is simply
```{math}
W_{app} = -\mu_0 \int_V \mathbf{H}_{app}\cdot\mathbf{M}d^3\mathbf{r}
```

## States, energies and solver requirements

The local minima of the magnetic energy depends on the trajectory by which it is reached, meaning we can match the representation of hysteresis in the calculation that matches
the hysteresis of magnetization observed in physical magnetic materials. Beign able to store in the magnetic state of a material a record of the history of its environment is 
precisely what makes magnetic materials interesting for store technologies.

The micromagnetic model is capable of representing and computing the major hysteresis loop of a material which reaches saturation magnetization when exposed to an strong enough 
magnetic field. A validity check that a micromagnetic model properly represents a physical material is to see wether computed values fo remanence and coercivity match with physical
measurements.

## Some obstacles to clear communication

1. Historically the study of micromagnetism has been done in multiple system of units, most notably Gaussian system and SI. This makes it challenging since the difference is not 
just a reescaling, but the fundamental relations change $\mathbf{B} = \mu_0 (\mathbf{M}+\mathbf{H})$ (SI) vs $\mathbf{B} = \mathbf{H}+4\pi\mathbf{M})$ (Gaussian)GaussiaGaussian
2. Another source of pontetial disagreement lies on the computation of the solutions itself due to float-point number operation carried by the algorithm.
3. Implicit assumption of initial conditions.
4. Symmetry breaking, sometimes symmetric states devolve into saddle points in the energy surface, meaning numerical imprecisions can push these states in different states.
5. Discretization can introduce "divots" which transform saddle points into false minima.
6. Programming errors.

## Tour of $\mu\text{MAG}$ Standard Problems

### 1. Hysteresis














