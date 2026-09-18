# PlautBeamColumnElement.jl

A beam-column finite element for the flexural-torsional deformation of imperfect,
braced thin-walled members under axial compression and transverse load,
implementing the governing equations of:

> Plaut, R.H., Moen, C.D. (2020). "Flexural-torsional deformations of imperfect
> thin-walled columns with discrete bracing." *Thin-Walled Structures*, 154, 106897.
> https://doi.org/10.1016/j.tws.2020.106897

together with the transverse-load and continuous-bracing terms of the companion beam
papers (Plaut & Moen 2019 SSRC; Plaut & Moen 2021 *JCSR* 179, 106534). It is the
finite element counterpart of
[ThinWalledBeamColumn.jl](https://github.com/runtosolve/ThinWalledBeamColumn.jl)
(finite differences) and the beam-column extension of
[PlautBeamFiniteElement.jl](https://github.com/runtosolve/PlautBeamFiniteElement.jl).

## Formulation

Coordinates follow Fig. 1 of the paper: x horizontal, y downward, z along the member,
twist ϕ positive clockwise. The centroid is at the origin, the shear center at
(`xo`, `yo`). `u`, `v` are the additional shear-center deflections and `ϕ` the
additional twist measured from the initial imperfect shape `u0`, `v0`, `ϕ0`. With
`eu = yo - hy`, `ev = hx - xo` the lever arms of a spring located at (`hx`, `hy`) from
the centroid, the equilibrium equations are

```
EIy u'''' + EIxy v'''' + kx (u + eu ϕ) + [P (u + u0)' + P yo (ϕ + ϕ0)']'          = qx
EIx v'''' + EIxy u'''' + ky (v + ev ϕ) + [P (v + v0)' - P xo (ϕ + ϕ0)']'          = qy
ECw ϕ'''' - GJ ϕ'' + kx eu (u + eu ϕ) + ky ev (v + ev ϕ) + kϕ ϕ
    + [P (Io/A) (ϕ + ϕ0)' + P yo (u + u0)' - P xo (v + v0)']'
    + (qx ax - qy ay) (ϕ + ϕ0)                                                   = qx ay + qy ax
```

For uniform P these are Eqs. (1)–(3) of the paper with the right-hand sides moved to
the left, the spring terms of Eqs. (4)–(6) written for continuous bracing, and the
load terms of JCSR Eqs. (2)–(4). P is compression positive and acts along the
centroids. Elastic stiffness and bracing act on the additional deformation; all
second-order (geometric) terms act on the total deformation.

The three fields are interpolated with cubic Hermite polynomials on two-node
elements with six DOFs per node (`u, u', v, v', ϕ, ϕ'`). The discrete equilibrium is

```
[Ke + Kg(P, q)] d = F(q) - Kg(P, q) d0
```

* **Elastic stiffness** `elastic_stiffness(model)`: bending (`EIy`, `EIx`, `EIxy`),
  warping (`ECw`) and St. Venant (`GJ`) torsion, continuous bracing `kx`, `ky`, `kϕ`
  with their lever arms, and discrete braces as point springs at nodes.
* **Axial geometric stiffness** `axial_geometric_stiffness(model)`: the `∫ P N'ᵀN'`
  kernels on u, v, ϕ (with `Io/A`) and the shear-center coupling `P yo`, `P xo`.
  P may vary along the member (linear within each element).
* **Load-position geometric stiffness** `load_geometric_stiffness(model)`:
  `(qx ax - qy ay) ϕ`, softening for gravity load above the shear center.
* **Imperfections**: half-sine amplitudes (paper Eq. 10) or arbitrary functions,
  represented by their Hermite interpolant `d0` and loading the member through
  `-Kg d0`.
* **Solution**: `R(d) = K d - F` solved with
  [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) (Newton–Raphson,
  diagonally equilibrated residual). A warning is raised when the tangent stiffness is
  indefinite (loads beyond a critical level).
* **Critical loads** `critical_load_factors(model)`: generalized eigenvalue problem
  `[Ke + Kq] d = λ (-KP) d`, returning load factors on the reference distribution `P`
  and mode shapes.

End conditions are `:pinned` (u = v = ϕ = 0, free to warp), `:fixed` (all six DOFs)
or `:free`. `Support(z; u, v, ϕ, u′, v′, ϕ′)` fixes individual DOFs at any node, so
frame connections that leave the section free to rotate, intermediate bridging,
multi-span members and cantilevers are all available.

## Usage

Consistent N–mm units: `E`, `G`, continuous `kx`, `ky` in N/mm²; continuous `kϕ` in
N/rad; discrete `kx`, `ky` in N/mm and `kϕ` in N·mm/rad; `A` in mm²; `I`, `J`, `Io`
in mm⁴; `Cw` in mm⁶; `P` in N; `qx`, `qy` in N/mm.

```julia
using PlautBeamColumnElement

# Example 1 of the paper: 362S162-54 lipped C stud, L = 2438 mm, pinned ends,
# discrete torsional brace at midheight, P = 10 kN
section  = Section(A = 272.0, Ix = 363370.0, Iy = 64100.0, J = 188.0, Cw = 122720891.0,
                   xo = -32.59, yo = 0.0)          # Io = Ix + Iy + A(xo² + yo²) by default
material = Material(E = 200e3)                    # G = E/2.6 by default
imperf   = Imperfection(u0 = 2438/1000, v0 = 2438/1000, ϕ0 = 0.00766)   # a1, a2, a3

model = BeamColumnModel(L = 2438.0, nel = 40, section = section, material = material,
                        P = 10e3, imperfection = imperf,
                        braces = [Brace(2438.0 / 2; kϕ = 230.7e3)])

sol = solve_beam_column(model)
twist(sol, 1219.0)                 # additional midheight twist ≈ 0.0075 rad (= a3, Section 3.2)
twist(sol, 1219.0; total = true)   # total twist ϕ0 + ϕ
deflections(sol, 1219.0)           # (u, v, ϕ)
brace_forces(sol)                  # (z, Fx, Fy, M) in each brace

# Example 2: W18×35 with a lateral brace at the bottom flange (hy = 225 mm)
I_section = Section(A = 6650.0, Ix = 2.12e8, Iy = 6.37e6, Io = 2.19e8, J = 2.11e5, Cw = 3.06e11)
col = BeamColumnModel(L = 2438.0, nel = 40, section = I_section, material = material, P = 1.0,
                      braces = [Brace(1219.0; kx = 20e3, hy = 225.0)])
critical_load_factors(col).factors[1]   # ≈ 2.67e6 N (Fig. 11); 2115 kN unbraced
```

Continuous bracing, transverse loads and a varying axial force, as used for the
free-flange model of a purlin line:

```julia
model = BeamColumnModel(z = collect(0.0:100.0:7620.0),   # any strictly increasing node set
                        section = flange, material = material,
                        restraint = Restraint(kx = 0.02, kϕ = 0.5, hx = 0.0, hy = 0.0),
                        qx = z -> shear_flow(z),          # scalar, (q1, q2), nodal vector or function
                        P = Pf,                            # nodal vector from the moment diagram
                        bc_left = :free, bc_right = :free,
                        supports = [Support(0.0; ϕ = false), Support(3810.0; v = false),
                                    Support(7620.0; ϕ = false)])
```

Section actions at the element ends are recovered by the matrix equilibrium method
`g = ke dₑ + kg (dₑ + d0ₑ) - fₑ`:

```julia
forces = element_end_forces(sol)   # one entry per element
forces[1].end1                     # (z, Vx, My, Vy, Mx, T, B) on the +z face at the left end
```

`Vx`, `Vy` include the transverse component of P (reactions come out directly), `My`
and `Mx` are second-order bending moments, `T` the total torque and `B` the bimoment.

## Relationship to ThinWalledBeamColumn.jl

The same equations and sign conventions (`ax`, `ay`, `qx`, `qy`, `P`, `kx`, `ky`,
`kϕ`) are used, so results agree for the cases both packages cover. Differences:

* **Shear-center offset.** The `P yo`, `P xo` coupling terms of Eqs. (1)–(3) are
  included; ThinWalledBeamColumn.jl omits them (it has no `xo`, `yo` inputs). For the
  362S162-54 stud this coupling lowers the pinned-pinned critical load from
  21.0 kN (pure torsion) to 19.5 kN (flexural-torsional). Set `xo = yo = 0` to recover
  ThinWalledBeamColumn.jl behaviour.
* **Brace location.** `hx`, `hy` are measured from the centroid (paper Fig. 1);
  ThinWalledBeamColumn.jl effectively measures them from the shear center. They agree
  when `xo = yo = 0`.
* **Varying P.** The weak form gives the conservative `(P u')'`, whereas
  ThinWalledBeamColumn.jl discretizes `P u''`. They coincide for uniform P.
* **Additions**: initial imperfections, discrete braces, `Ixy` coupling, critical
  load factors, per-DOF supports, element end forces and a stability flag.

## Verification

`test/runtests.jl` checks the element against closed-form and paper results:

* exact secant-formula beam-column response (deflection, second-order moment,
  reactions) and Euler, torsional, flexural-torsional (C stud, xo ≠ 0), fixed-fixed,
  flagpole and mid-supported critical loads;
* exact sine-series solutions of the full coupled equations for pinned-pinned members
  with continuous restraints, shear-center offsets, transverse loads with offsets,
  uniform P and half-sine imperfections (Examples 1 and 2 sections);
* imperfection amplification `1/(1 - P/Pcr)` for half-sine and full-sine shapes;
* paper Example 1: kϕ = 230.7 kN·mm/rad at P = 10 kN gives an additional twist equal
  to the initial twist; peak total twist 0.021 rad at z = 1337 mm for α = 0.2,
  kϕ = 500 kN·mm/rad; the trends of Figs. 3–6;
* paper Example 2: critical load 2115 kN unbraced and ≈ 2670 kN with kx = 20 kN/mm
  (Fig. 11); Fig. 12 twists; peak total twist 0.019 rad at z = 1185 mm for α = 0.2,
  P = 1500 kN, kx = 100 kN/mm (Fig. 16); the trends of Figs. 13–14;
* JCSR beam terms via `hy = -ay` (P = 0), rigid-body twist against a continuous
  rotational spring, linearly varying P against a sine-basis Galerkin reference;
* element end force continuity, brace-force and support-reaction jumps, bimoment
  boundary conditions; two-span/pinned-fixed equivalence; non-uniform meshes;
  per-element properties; mesh convergence; input validation.
