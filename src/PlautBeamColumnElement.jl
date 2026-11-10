"""
    PlautBeamColumnElement

Finite element implementation of the flexural-torsional deformation equations for
thin-walled beam-columns with initial geometric imperfections and continuous or
discrete elastic bracing, from:

    Plaut, R.H., Moen, C.D. (2020). "Flexural-torsional deformations of imperfect
    thin-walled columns with discrete bracing." Thin-Walled Structures, 154, 106897.
    https://doi.org/10.1016/j.tws.2020.106897

combined with the transverse-load and continuous-bracing terms of the companion
beam papers (Plaut & Moen 2019, SSRC Annual Stability Conference; Plaut & Moen
2021, JCSR 179, 106534). These are the equations solved by finite differences in
ThinWalledBeamColumn.jl, extended here with the shear-center offset (xo, yo)
coupling terms of the 2020 paper, initial imperfections, discrete braces, and a
critical-load eigenvalue solver.

Coordinates follow Fig. 1 of the 2020 paper: x horizontal, y downward, z along the
column, twist ϕ positive clockwise (a positive rotation about +z). The centroid is
at the origin and the shear center at (xo, yo). `u`, `v` are the additional
deflections of the shear center and `ϕ` the additional twist, measured from the
initial imperfect shape `u0`, `v0`, `ϕ0`. With primes denoting d/dz and
`eu = yo - hy`, `ev = hx - xo` (lever arms of a spring located at (hx, hy)), the
equilibrium equations are

    EIy u'''' + EIxy v'''' + kx (u + eu ϕ)
        + [P (u + u0)' + P yo (ϕ + ϕ0)']' - (Mx (ϕ + ϕ0))''                  = qx
    EIx v'''' + EIxy u'''' + ky (v + ev ϕ)
        + [P (v + v0)' - P xo (ϕ + ϕ0)']' + (My (ϕ + ϕ0))''                  = qy
    ECw ϕ'''' - GJ ϕ'' + kx eu (u + eu ϕ) + ky ev (v + ev ϕ) + kϕ ϕ
        + [P (Io/A) (ϕ + ϕ0)' + P yo (u + u0)' - P xo (v + v0)']'
        + (qx ax - qy ay) (ϕ + ϕ0) - Mx (u + u0)'' + My (v + v0)''          = qx ay + qy ax

For uniform P these are Eqs. (1)-(3) of the 2020 paper (right-hand sides moved to
the left), with the spring terms of its Eqs. (4)-(6) written for continuous bracing
and the load terms of JCSR Eqs. (2)-(4), extended (v0.4.0) with the classical
lateral-torsional couplings between the first-order bending moments and the twist,
the Mx, My terms, as in PlautBeamElement.jl v0.5.0. Mx = E (Ix v'' + Ixy u'') and
My = E (Iy u'' + Ixy v'') are the first-order moments of the transverse loads (from a
linear solve with the elastic stiffness alone, so continuous spans, fixed ends and
bracing reactions are included; y downward, so a sagging Mx is negative). They come
from the second-order strain energy  U₂ = ∫ (-Mx ϕ u'' + My ϕ v'') dz : twist turns
part of Mx into a weak-axis moment, and lateral curvature turns part of Mx into a
torque. The coupling is on by default; `BeamColumnModel(...; moment_coupling =
false)` recovers the previous equations exactly. P is compression positive and acts
along the centroids. Elastic stiffness and bracing act on the additional
deformation only; all second-order (geometric) terms act on the total deformation,
so the imperfection enters the discrete equations as the load  -Kg d0.

The three fields are interpolated with cubic Hermite polynomials on two-node
elements; each node carries six degrees of freedom (u, u', v, v', ϕ, ϕ'). The
discrete equilibrium is

    [Ke + Kg(P, q)] d = F(q) - Kg(P, q) d0

with Ke the elastic + bracing stiffness and Kg the geometric stiffness from the
axial force (∫ P N'ᵀN' kernels), from the load position ((qx ax - qy ay) ϕ) and from
the moment coupling (the symmetric blocks Kg[uϕ] = -∫ Mx N''ᵀN dz,
Kg[vϕ] = ∫ My N''ᵀN dz and their transposes). Kg depends on the loads (P, q) but not
on the displacements d, so for a given load state this is a linear system and is
solved directly; the response is nonlinear in the load magnitude only because Kg
scales with the load.
Discrete braces are added as point springs at nodes. Property and load variation
along the length is allowed: section, material and continuous restraint per
element; P, qx, qy at nodes (linear inside each element).

Consistent units are expected, e.g. N and mm: E, G in N/mm²; A in mm²; I, J, Io in
mm⁴; Cw in mm⁶; P in N; qx, qy in N/mm; continuous kx, ky in N/mm² and kϕ in N/rad;
discrete kx, ky in N/mm and kϕ in N·mm/rad.
"""
module PlautBeamColumnElement

using LinearAlgebra
using SparseArrays

export Section, Material, Restraint, Brace, Support, Imperfection
export BeamColumnModel, BeamColumnSolution
export nodes, elastic_stiffness, geometric_stiffness, axial_geometric_stiffness,
       load_geometric_stiffness, moment_coupling_stiffness, sparse_elastic_stiffness,
       sparse_geometric_stiffness, load_vector, imperfection_vector, constrained_dofs
export solve_beam_column, deflections, twist, lateral_deflection, vertical_deflection,
       initial_deflections, total_deflections
export element_end_forces, brace_forces, critical_load_factors, critical_transverse_load_factor,
       prebuckling_moments, PrebucklingMoments
export Inputs, Outputs, Model

# ------------------------------------------------------------------
# Input types
# ------------------------------------------------------------------

"""
    Section(; A, Ix, Iy, Ixy=0.0, J, Cw, xo=0.0, yo=0.0, Io=Ix+Iy+A(xo²+yo²), ax=0.0, ay=0.0)

Cross-section properties in centroidal x-y axes (x horizontal, y downward).

  * `A` area; `Ix`, `Iy`, `Ixy` moments of inertia about the centroid (principal
    axes are not assumed: `Ixy` couples the two bending equations); `J` torsion
    constant; `Cw` warping constant.
  * `xo`, `yo` shear center location relative to the centroid (x0, y0 in the 2020
    paper). They couple bending and twist under axial load.
  * `Io` polar moment of inertia about the shear center (I0 in the paper); defaults
    to `Ix + Iy + A (xo² + yo²)`.
  * `ax`, `ay` locate the point of application of the transverse loads qx, qy
    relative to the shear center, with the convention of the JCSR beam paper:
    `ax` along +x, `ay` upward (opposite to +y), so a load on the top flange has
    `ay > 0`.
"""
struct Section
    A::Float64
    Ix::Float64
    Iy::Float64
    Ixy::Float64
    J::Float64
    Cw::Float64
    xo::Float64
    yo::Float64
    Io::Float64
    ax::Float64
    ay::Float64
end
function Section(; A, Ix, Iy, Ixy = 0.0, J, Cw, xo = 0.0, yo = 0.0,
                 Io = Ix + Iy + A * (xo^2 + yo^2), ax = 0.0, ay = 0.0)
    A > 0 || throw(ArgumentError("A must be positive, got $A"))
    Io > 0 || throw(ArgumentError("Io must be positive, got $Io"))
    Section(A, Ix, Iy, Ixy, J, Cw, xo, yo, Io, ax, ay)
end

"""
    Material(; E, G=E/2.6)

Linearly elastic material. The papers use G = E/2.6 for steel.
"""
struct Material
    E::Float64
    G::Float64
end
Material(; E, G = E / 2.6) = Material(E, G)

"""
    Restraint(; kx=0.0, ky=0.0, kϕ=0.0, hx=0.0, hy=0.0)

Continuous elastic bracing per unit length: `kx`, `ky` translational stiffnesses
parallel to x and y acting at the point (`hx`, `hy`) measured from the centroid
(x right, y down, as N = (hx, hy) in Fig. 1 of the 2020 paper), and `kϕ` a
rotational stiffness. The spring displacements are

    u_N = u + (yo - hy) ϕ,     v_N = v + (hx - xo) ϕ

so for a lateral spring on the top flange of a section whose shear center is at
the centroid, `hy = -ay` reproduces the JCSR term kx (u + ay ϕ). In
ThinWalledBeamColumn.jl the same quantities are effectively measured from the
shear center; the two agree when `xo = yo = 0`.
"""
struct Restraint
    kx::Float64
    ky::Float64
    kϕ::Float64
    hx::Float64
    hy::Float64
end
Restraint(; kx = 0.0, ky = 0.0, kϕ = 0.0, hx = 0.0, hy = 0.0) = Restraint(kx, ky, kϕ, hx, hy)

"""
    Brace(z; kx=0.0, ky=0.0, kϕ=0.0, hx=0.0, hy=0.0)

Discrete elastic brace at location `z` (which must coincide with a mesh node), the
restraint of Eqs. (4)-(6) of the 2020 paper: translational stiffnesses `kx`, `ky`
(force/length) acting at (`hx`, `hy`) from the centroid and rotational stiffness
`kϕ` (moment/rad). The brace resists the additional deformation only.
"""
struct Brace
    z::Float64
    kx::Float64
    ky::Float64
    kϕ::Float64
    hx::Float64
    hy::Float64
end
Brace(z; kx = 0.0, ky = 0.0, kϕ = 0.0, hx = 0.0, hy = 0.0) = Brace(z, kx, ky, kϕ, hx, hy)

"""
    Support(z; u=true, v=true, ϕ=true, u′=false, v′=false, ϕ′=false)

Rigid support at location `z` (a mesh node). Each flag fixes the corresponding
degree of freedom to zero: translations `u`, `v`, twist `ϕ`, rotations `u′`, `v′`,
and warping `ϕ′`. The default fixes u, v, ϕ (a pin free to warp). For example, a
frame connection through the bottom flange that leaves the section free to rotate
is `Support(z; ϕ = false)`, and intermediate bridging that restrains lateral
movement and twist but not vertical deflection is `Support(z; v = false)`.
"""
struct Support
    z::Float64
    u::Bool
    v::Bool
    ϕ::Bool
    u′::Bool
    v′::Bool
    ϕ′::Bool
end
Support(z; u = true, v = true, ϕ = true, u′ = false, v′ = false, ϕ′ = false) =
    Support(z, u, v, ϕ, u′, v′, ϕ′)

"""
    Imperfection(; u0=0.0, v0=0.0, ϕ0=0.0)

Initial geometric imperfection of the unloaded column: lateral deflection `u0`,
vertical deflection `v0` (both of the shear center) and twist `ϕ0`. Each entry is
either

  * a number — the amplitude of a half-sine shape over the member length,
    `a sin(π (z - z₁)/L)`, Eq. (10) of the 2020 paper (a1, a2, a3), or
  * a function `z -> value`.

The imperfection is represented by its Hermite interpolant on the mesh and enters
the equations through the geometric stiffness only (paper Eqs. 1-3, right-hand
sides).
"""
struct Imperfection
    u0::Union{Float64, Function}
    v0::Union{Float64, Function}
    ϕ0::Union{Float64, Function}
end
Imperfection(; u0 = 0.0, v0 = 0.0, ϕ0 = 0.0) =
    Imperfection(_imp(u0), _imp(v0), _imp(ϕ0))
_imp(x::Real) = Float64(x)
_imp(f::Function) = f

"""
    BeamColumnModel(; L, nel, section, material, kwargs...)
    BeamColumnModel(; z, section, material, kwargs...)

Thin-walled beam-column meshed either uniformly (`L` and `nel` equal elements) or
on the node coordinates `z` (strictly increasing; the mesh may be non-uniform).

Keyword arguments:

  * `section::Section`, `material::Material`, `restraint::Restraint` — a single
    value for a prismatic member, or a vector with one entry per element.
  * `braces::Vector{Brace}` — discrete springs at nodes.
  * `supports::Vector{Support}` — rigid supports at nodes (any node, including the
    ends), in addition to the end conditions.
  * `bc_left`, `bc_right` — end conditions at `z₁` and `z_end`:
    `:pinned` (u = v = ϕ = 0, free to warp and to rotate), `:fixed` (all six DOFs
    zero) or `:free` (no constraint, e.g. a cantilever tip).
  * `qx`, `qy` — transverse loads per unit length along +x and +y (+y is down, so
    gravity is `qy > 0`), applied at (`ax`, `ay`) of the section.
  * `P` — axial force, compression positive, acting along the centroids.
  * `imperfection::Imperfection`.
  * `moment_coupling` (default `true`) — include the lateral-torsional coupling
    between the first-order moments of the transverse loads and the twist in the
    geometric stiffness (module documentation). With `false` the geometric
    stiffness has the axial-force and load-height terms only.

`qx`, `qy` and `P` may each be a scalar (uniform), a 2-tuple of the values at the
two ends (linear variation), a vector of nodal values (linear within each element)
or a function of `z`.
"""
struct BeamColumnModel
    z::Vector{Float64}
    sections::Vector{Section}
    materials::Vector{Material}
    restraints::Vector{Restraint}
    braces::Vector{Brace}
    supports::Vector{Support}
    imperfection::Imperfection
    qx::Vector{Float64}
    qy::Vector{Float64}
    P::Vector{Float64}
    bc_left::Symbol
    bc_right::Symbol
    moment_coupling::Bool
end

function BeamColumnModel(; L = nothing, nel = nothing, z = nothing,
                         section, material, restraint = Restraint(),
                         braces = Brace[], supports = Support[],
                         imperfection = Imperfection(),
                         qx = 0.0, qy = 0.0, P = 0.0,
                         bc_left = :pinned, bc_right = :pinned,
                         moment_coupling::Bool = true)
    if z === nothing
        (L === nothing || nel === nothing) &&
            throw(ArgumentError("give either node coordinates z or both L and nel"))
        nel >= 2 || throw(ArgumentError("need at least 2 elements"))
        zz = collect(range(0.0, Float64(L), length = Int(nel) + 1))
    else
        (L === nothing && nel === nothing) ||
            throw(ArgumentError("give either node coordinates z or L and nel, not both"))
        zz = Float64.(collect(z))
        length(zz) >= 3 || throw(ArgumentError("need at least 3 nodes (2 elements)"))
        all(diff(zz) .> 0) || throw(ArgumentError("z must be strictly increasing"))
    end
    ne = length(zz) - 1
    for bc in (bc_left, bc_right)
        bc in (:pinned, :fixed, :free) ||
            throw(ArgumentError("end condition must be :pinned, :fixed or :free, got $bc"))
    end
    model = BeamColumnModel(zz,
                            per_element(section, ne, Section),
                            per_element(material, ne, Material),
                            per_element(restraint, ne, Restraint),
                            collect(Brace, braces), collect(Support, supports),
                            imperfection,
                            nodal_values(qx, zz), nodal_values(qy, zz), nodal_values(P, zz),
                            bc_left, bc_right, moment_coupling)
    for b in model.braces
        node_at(model, b.z)          # validate brace locations
    end
    for s in model.supports
        node_at(model, s.z)          # validate support locations
    end
    return model
end

per_element(x::T, ne, ::Type{T}) where {T} = fill(x, ne)
function per_element(x::AbstractVector, ne, ::Type{T}) where {T}
    length(x) == ne ||
        throw(ArgumentError("expected one $(nameof(T)) per element ($ne), got $(length(x))"))
    return collect(T, x)
end

nodal_values(x::Real, z) = fill(Float64(x), length(z))
nodal_values(x::Function, z) = Float64[x(zi) for zi in z]
function nodal_values(x::Union{Tuple, AbstractVector}, z)
    n = length(z)
    if length(x) == 2 && n != 2
        return Float64[x[1] + (x[2] - x[1]) * (zi - z[1]) / (z[end] - z[1]) for zi in z]
    end
    length(x) == n ||
        throw(ArgumentError("expected a scalar, two end values, or $n nodal values, got $(length(x))"))
    return Float64.(collect(x))
end

"Node coordinates of the mesh."
nodes(m::BeamColumnModel) = m.z
nnodes(m::BeamColumnModel) = length(m.z)
nelements(m::BeamColumnModel) = length(m.z) - 1
ndofs(m::BeamColumnModel) = 6 * nnodes(m)
span(m::BeamColumnModel) = m.z[end] - m.z[1]
elength(m::BeamColumnModel, e::Int) = m.z[e + 1] - m.z[e]

"Index of the mesh node at coordinate `zs` (error if none coincides)."
function node_at(m::BeamColumnModel, zs::Real)
    i = argmin(abs.(m.z .- zs))
    isapprox(m.z[i], zs; atol = 1e-8 * span(m)) ||
        throw(ArgumentError("location z = $zs does not coincide with a mesh node"))
    return i
end

"Lever arms (eu, ev) of a spring at (hx, hy) from the centroid: u_N = u + eu ϕ, v_N = v + ev ϕ."
lever_arms(s::Section, hx::Real, hy::Real) = (s.yo - hy, hx - s.xo)

"3×3 spring kernel on (u, v, ϕ) for stiffnesses (kx, ky, kϕ) with lever arms (eu, ev)."
function spring_kernel(kx, ky, kϕ, eu, ev)
    return [kx        0.0       kx * eu
            0.0       ky        ky * ev
            kx * eu   ky * ev   kx * eu^2 + ky * ev^2 + kϕ]
end

# ------------------------------------------------------------------
# Hermite cubic element matrices on [0, Le], DOF order (w1, w1', w2, w2')
# ------------------------------------------------------------------

function hermite(ξ::Float64, Le::Float64)
    (1 - 3ξ^2 + 2ξ^3,
     Le * (ξ - 2ξ^2 + ξ^3),
     3ξ^2 - 2ξ^3,
     Le * (ξ^3 - ξ^2))
end

"d/dz of the Hermite shape functions."
function hermite_d(ξ::Float64, Le::Float64)
    ((-6ξ + 6ξ^2) / Le,
     1 - 4ξ + 3ξ^2,
     (6ξ - 6ξ^2) / Le,
     3ξ^2 - 2ξ)
end

"d²/dz² of the Hermite shape functions."
hermite_dd(ξ::Float64, Le::Float64) =
    ((12ξ - 6) / Le^2, (6ξ - 4) / Le, (6 - 12ξ) / Le^2, (6ξ - 2) / Le)

"∫ N''ᵀ N'' dz — bending/warping stiffness kernel."
k_bend(Le) = (1 / Le^3) * [ 12.0    6Le   -12.0    6Le
                             6Le   4Le^2   -6Le   2Le^2
                           -12.0   -6Le    12.0   -6Le
                             6Le   2Le^2   -6Le   4Le^2 ]

"∫ N'ᵀ N' dz — St. Venant torsion / axial-force kernel."
k_grad(Le) = (1 / (30Le)) * [ 36.0    3Le   -36.0    3Le
                               3Le   4Le^2   -3Le   -Le^2
                             -36.0   -3Le    36.0   -3Le
                               3Le   -Le^2   -3Le   4Le^2 ]

"∫ Nᵀ N dz — elastic foundation / load-position kernel."
k_found(Le) = (Le / 420) * [ 156.0   22Le    54.0  -13Le
                              22Le   4Le^2   13Le  -3Le^2
                              54.0   13Le   156.0  -22Le
                             -13Le  -3Le^2  -22Le   4Le^2 ]

# 4-point Gauss-Legendre rule on ξ ∈ [0, 1]: exact for polynomials of degree ≤ 7,
# which covers a linear weight times Nᵢ Nⱼ (degree 7), Nᵢ' Nⱼ' (degree 5) and Nᵢ (degree 4).
const GAUSS4 = let a = sqrt(3 / 7 - 2 / 7 * sqrt(6 / 5)), b = sqrt(3 / 7 + 2 / 7 * sqrt(6 / 5)),
                   wa = (18 + sqrt(30)) / 72, wb = (18 - sqrt(30)) / 72
    ((ξ = (1 - b) / 2, w = wb), (ξ = (1 - a) / 2, w = wa),
     (ξ = (1 + a) / 2, w = wa), (ξ = (1 + b) / 2, w = wb))
end

"∫ w(ξ) Nᵀ N dz with w varying linearly from `w1` (ξ=0) to `w2` (ξ=1) — exact."
function k_found_linear(Le, w1, w2)
    K = zeros(4, 4)
    for gp in GAUSS4
        N = collect(hermite(gp.ξ, Le))
        K .+= ((w1 * (1 - gp.ξ) + w2 * gp.ξ) * gp.w * Le) .* (N * N')
    end
    return K
end

"∫ w(ξ) N'ᵀ N' dz with w varying linearly from `w1` to `w2` — exact."
function k_grad_linear(Le, w1, w2)
    K = zeros(4, 4)
    for gp in GAUSS4
        B = collect(hermite_d(gp.ξ, Le))
        K .+= ((w1 * (1 - gp.ξ) + w2 * gp.ξ) * gp.w * Le) .* (B * B')
    end
    return K
end

"∫ w(ξ) Nᵀ dz — consistent nodal loads for a load varying linearly from `w1` to `w2`."
function f_linear(Le, w1, w2)
    f = zeros(4)
    for gp in GAUSS4
        N = collect(hermite(gp.ξ, Le))
        f .+= ((w1 * (1 - gp.ξ) + w2 * gp.ξ) * gp.w * Le) .* N
    end
    return f
end

# Local DOF indices of each field within the 12-DOF element (u, u', v, v', ϕ, ϕ' per node)
const UDOF = (1, 2, 7, 8)
const VDOF = (3, 4, 9, 10)
const ΦDOF = (5, 6, 11, 12)
const FIELDS = (UDOF, VDOF, ΦDOF)

function add_block!(ke::Matrix{Float64}, rows, cols, coeff::Float64, k4::Matrix{Float64})
    iszero(coeff) && return ke
    @inbounds for j in 1:4, i in 1:4
        ke[rows[i], cols[j]] += coeff * k4[i, j]
    end
    return ke
end

"Add a 3×3 (u, v, ϕ) kernel `k3` weighted by the 4×4 shape-function kernel `k4`."
function add_kernel!(ke::Matrix{Float64}, k3::AbstractMatrix, k4::Matrix{Float64})
    for a in 1:3, b in 1:3
        add_block!(ke, FIELDS[a], FIELDS[b], Float64(k3[a, b]), k4)
    end
    return ke
end

"""
    elastic_element_stiffness(model, e)

12×12 elastic stiffness matrix of element `e`: bending (EIy, EIx, EIxy), warping
(ECw) and St. Venant (GJ) torsion, and the continuous bracing terms kx, ky, kϕ
with their lever arms about the shear center.
"""
function elastic_element_stiffness(m::BeamColumnModel, e::Int)
    (; E, G) = m.materials[e]
    s = m.sections[e]
    r = m.restraints[e]
    Le = elength(m, e)
    kb, kg, kf = k_bend(Le), k_grad(Le), k_found(Le)

    ke = zeros(12, 12)
    add_block!(ke, UDOF, UDOF, E * s.Iy, kb)              # EIy u''''
    add_block!(ke, UDOF, VDOF, E * s.Ixy, kb)             # EIxy v''''
    add_block!(ke, VDOF, UDOF, E * s.Ixy, kb)             # EIxy u''''
    add_block!(ke, VDOF, VDOF, E * s.Ix, kb)              # EIx v''''
    add_block!(ke, ΦDOF, ΦDOF, E * s.Cw, kb)              # ECw ϕ''''
    add_block!(ke, ΦDOF, ΦDOF, G * s.J, kg)               # -GJ ϕ''
    eu, ev = lever_arms(s, r.hx, r.hy)
    add_kernel!(ke, spring_kernel(r.kx, r.ky, r.kϕ, eu, ev), kf)   # continuous bracing
    return ke
end

"""
    axial_geometric_element_stiffness(model, e)

12×12 geometric stiffness of element `e` from the axial force P (compression
positive), varying linearly between the nodal values. From the weak form of the
terms [P u']', [P v']', [P (Io/A) ϕ']' and the shear-center coupling
[P yo ϕ']', [P yo u']', -[P xo ϕ']', -[P xo v']' of Eqs. (1)-(3):

    kP = -∫ P N'ᵀ N' dz ⊗ [ 1    0    yo
                             0    1   -xo
                             yo  -xo   Io/A ]
"""
function axial_geometric_element_stiffness(m::BeamColumnModel, e::Int)
    s = m.sections[e]
    P1, P2 = m.P[e], m.P[e + 1]
    ke = zeros(12, 12)
    (iszero(P1) && iszero(P2)) && return ke
    kg = k_grad_linear(elength(m, e), P1, P2)
    c = [ 1.0    0.0    s.yo
          0.0    1.0   -s.xo
          s.yo  -s.xo   s.Io / s.A ]
    add_kernel!(ke, -c, kg)
    return ke
end

"""
    load_geometric_element_stiffness(model, e)

12×12 geometric (load-position) stiffness of element `e`: the twist-dependent
torque of the transverse loads, (qx ax - qy ay) ϕ, softening for gravity load
applied above the shear center and stiffening for uplift (JCSR Eq. 4).
"""
function load_geometric_element_stiffness(m::BeamColumnModel, e::Int)
    (; ax, ay) = m.sections[e]
    w1 = m.qx[e] * ax - m.qy[e] * ay
    w2 = m.qx[e + 1] * ax - m.qy[e + 1] * ay
    ke = zeros(12, 12)
    (iszero(w1) && iszero(w2)) && return ke
    add_block!(ke, ΦDOF, ΦDOF, 1.0, k_found_linear(elength(m, e), w1, w2))
    return ke
end

"""
    PrebucklingMoments

First-order moments of the transverse loads on one element, from a linear solve with
the elastic stiffness alone: end values `Mx1`, `Mx2`, `My1`, `My2` (sign convention of
[`element_end_forces`](@ref): Mx = E (Ix v'' + Ixy u''), My = E (Iy u'' + Ixy v'')) and
the end values of the second derivatives `px = My'' = qx - kx (u + eu ϕ)` and
`py = Mx'' = qy - ky (v + ev ϕ)`, so that the moment inside the element is recovered
exactly (cubic) by [`moment_at`](@ref).
"""
struct PrebucklingMoments
    Mx1::Float64
    Mx2::Float64
    My1::Float64
    My2::Float64
    px1::Float64
    px2::Float64
    py1::Float64
    py2::Float64
end

"""
    moment_at(M1, M2, p1, p2, Le, ξ)

Moment at ξ ∈ [0, 1] on an element whose end moments are `M1`, `M2` and whose second
derivative M'' varies linearly from `p1` to `p2`: the linear interpolant plus the
particular solution that vanishes at both ends.
"""
function moment_at(M1, M2, p1, p2, Le, ξ)
    f1 = ξ^2 / 2 - ξ^3 / 6 - ξ / 3
    f2 = ξ^3 / 6 - ξ / 6
    return M1 * (1 - ξ) + M2 * ξ + Le^2 * (p1 * f1 + p2 * f2)
end

"""
    moment_coupling_element_stiffness(model, e, moments)

12×12 geometric stiffness of element `e` from the lateral-torsional coupling of the
first-order moments with the twist: the symmetric u-ϕ and v-ϕ blocks
-∫ Mx N''ᵀ N dz and ∫ My N''ᵀ N dz (and their transposes), from the second-order
energy ∫ (-Mx ϕ u'' + My ϕ v'') dz. `moments` is the vector returned by
[`prebuckling_moments`](@ref) (`nothing` gives a zero matrix). Integrated exactly
with 4-point Gauss quadrature (cubic moment × degree-4 integrand).
"""
function moment_coupling_element_stiffness(m::BeamColumnModel, e::Int, moments)
    ke = zeros(12, 12)
    moments === nothing && return ke
    M = moments[e]
    Le = elength(m, e)
    kuϕ = zeros(4, 4)
    kvϕ = zeros(4, 4)
    for gp in GAUSS4
        N = collect(hermite(gp.ξ, Le))
        B = collect(hermite_dd(gp.ξ, Le))
        Mx = moment_at(M.Mx1, M.Mx2, M.py1, M.py2, Le, gp.ξ)
        My = moment_at(M.My1, M.My2, M.px1, M.px2, Le, gp.ξ)
        w = gp.w * Le
        kuϕ .-= (Mx * w) .* (B * N')
        kvϕ .+= (My * w) .* (B * N')
    end
    add_block!(ke, UDOF, ΦDOF, 1.0, kuϕ)
    add_block!(ke, ΦDOF, UDOF, 1.0, permutedims(kuϕ))
    add_block!(ke, VDOF, ΦDOF, 1.0, kvϕ)
    add_block!(ke, ΦDOF, VDOF, 1.0, permutedims(kvϕ))
    return ke
end

"""
    geometric_element_stiffness(model, e[, moments])

Total geometric element stiffness kg = kP + kq + kM (axial force, load position and
moment coupling). `moments` are the pre-buckling moments of
[`prebuckling_moments`](@ref); without the argument they are computed when
`model.moment_coupling` is set (convenient, but it repeats a global first-order
solve for every call; the assembly routines pass them in).
"""
geometric_element_stiffness(m::BeamColumnModel, e::Int) =
    geometric_element_stiffness(m, e, prebuckling_moments(m))
geometric_element_stiffness(m::BeamColumnModel, e::Int, moments) =
    axial_geometric_element_stiffness(m, e) + load_geometric_element_stiffness(m, e) +
    moment_coupling_element_stiffness(m, e, moments)

"""
    element_load_vector(model, e)

12-component consistent load vector of element `e`: qx on u, qy on v, and the
ϕ-independent torque qx ay + qy ax on ϕ, each varying linearly over the element.
"""
function element_load_vector(m::BeamColumnModel, e::Int)
    (; ax, ay) = m.sections[e]
    Le = elength(m, e)
    qx1, qx2 = m.qx[e], m.qx[e + 1]
    qy1, qy2 = m.qy[e], m.qy[e + 1]
    fe = zeros(12)
    fe[collect(UDOF)] = f_linear(Le, qx1, qx2)
    fe[collect(VDOF)] = f_linear(Le, qy1, qy2)
    fe[collect(ΦDOF)] = f_linear(Le, qx1 * ay + qy1 * ax, qx2 * ay + qy2 * ax)
    return fe
end

# ------------------------------------------------------------------
# Global assembly
# ------------------------------------------------------------------

element_dofs(e::Int) = (6 * (e - 1)) .+ (1:12)
node_dofs(i::Int) = (6 * (i - 1)) .+ (1:6)

function assemble(m::BeamColumnModel, element_matrix::Function)
    K = zeros(ndofs(m), ndofs(m))
    for e in 1:nelements(m)
        dofs = element_dofs(e)
        @views K[dofs, dofs] .+= element_matrix(m, e)
    end
    return K
end

"""
    brace_stiffness(model)

Global stiffness contribution of the discrete braces: the 3×3 spring kernel of
each brace on the (u, v, ϕ) DOFs of its node, from Eqs. (4)-(6).
"""
function brace_stiffness(m::BeamColumnModel)
    K = zeros(ndofs(m), ndofs(m))
    for b in m.braces
        i = node_at(m, b.z)
        s = m.sections[min(i, nelements(m))]
        eu, ev = lever_arms(s, b.hx, b.hy)
        k3 = spring_kernel(b.kx, b.ky, b.kϕ, eu, ev)
        dofs = 6 * (i - 1) .+ [1, 3, 5]                    # u, v, ϕ
        @views K[dofs, dofs] .+= k3
    end
    return K
end

"Global elastic stiffness matrix (member + continuous bracing + discrete braces)."
elastic_stiffness(m::BeamColumnModel) =
    assemble(m, elastic_element_stiffness) + brace_stiffness(m)

"Global geometric stiffness from the axial force P (Eqs. 1-3)."
axial_geometric_stiffness(m::BeamColumnModel) = assemble(m, axial_geometric_element_stiffness)

"Global geometric stiffness from the position of the transverse loads (JCSR Eq. 4)."
load_geometric_stiffness(m::BeamColumnModel) = assemble(m, load_geometric_element_stiffness)

"""
    moment_coupling_stiffness(model; moments = prebuckling_moments(model))

Global geometric stiffness from the lateral-torsional coupling of the first-order
moments with the twist (zero when `model.moment_coupling` is false).
"""
moment_coupling_stiffness(m::BeamColumnModel; moments = prebuckling_moments(m)) =
    assemble(m, (mm, e) -> moment_coupling_element_stiffness(mm, e, moments))

"""
    geometric_stiffness(model; moments = prebuckling_moments(model))

Total global geometric stiffness Kg = KP + Kq + KM (axial force, load position,
moment coupling).
"""
geometric_stiffness(m::BeamColumnModel; moments = prebuckling_moments(m)) =
    assemble(m, (mm, e) -> geometric_element_stiffness(mm, e, moments))

# Sparse assembly: each element couples only its own 12 DOFs and each brace only
# the (u, v, ϕ) DOFs of one node, so K is banded and sparse storage grows linearly
# with the number of elements (the dense matrices above grow quadratically).

"Collect the (row, col, value) triplets of the element matrices `element_matrix(m, e)`."
function element_triplets!(I, J, V, m::BeamColumnModel, element_matrix::Function)
    for e in 1:nelements(m)
        dofs = element_dofs(e)
        ke = element_matrix(m, e)
        for c in 1:12, r in 1:12
            push!(I, dofs[r]); push!(J, dofs[c]); push!(V, ke[r, c])
        end
    end
    return I, J, V
end

"""
    sparse_elastic_stiffness(model)

Sparse version of [`elastic_stiffness`](@ref): member + continuous bracing +
discrete braces.
"""
function sparse_elastic_stiffness(m::BeamColumnModel)
    n = 144 * nelements(m) + 9 * length(m.braces)
    I, J, V = sizehint!(Int[], n), sizehint!(Int[], n), sizehint!(Float64[], n)
    element_triplets!(I, J, V, m, elastic_element_stiffness)
    for b in m.braces
        i = node_at(m, b.z)
        s = m.sections[min(i, nelements(m))]
        eu, ev = lever_arms(s, b.hx, b.hy)
        k3 = spring_kernel(b.kx, b.ky, b.kϕ, eu, ev)
        dofs = 6 * (i - 1) .+ (1, 3, 5)                    # u, v, ϕ
        for c in 1:3, r in 1:3
            push!(I, dofs[r]); push!(J, dofs[c]); push!(V, k3[r, c])
        end
    end
    return sparse(I, J, V, ndofs(m), ndofs(m))   # shared entries are summed
end

"""
    sparse_geometric_stiffness(model; moments = prebuckling_moments(model))

Sparse version of [`geometric_stiffness`](@ref): Kg = KP + Kq + KM.
"""
function sparse_geometric_stiffness(m::BeamColumnModel; moments = prebuckling_moments(m))
    n = 144 * nelements(m)
    I, J, V = sizehint!(Int[], n), sizehint!(Int[], n), sizehint!(Float64[], n)
    element_triplets!(I, J, V, m, (mm, e) -> geometric_element_stiffness(mm, e, moments))
    return sparse(I, J, V, ndofs(m), ndofs(m))
end

"Global consistent load vector of the transverse loads."
function load_vector(m::BeamColumnModel)
    F = zeros(ndofs(m))
    for e in 1:nelements(m)
        @views F[element_dofs(e)] .+= element_load_vector(m, e)
    end
    return F
end

# ------------------------------------------------------------------
# Imperfections
# ------------------------------------------------------------------

imperfection_value(a::Float64, z, z1, L) = a * sin(π * (z - z1) / L)
imperfection_slope(a::Float64, z, z1, L) = a * (π / L) * cos(π * (z - z1) / L)
imperfection_value(f::Function, z, z1, L) = Float64(f(z))

"Fourth-order finite-difference slope of a user imperfection function, kept inside [z1, z1+L]."
function imperfection_slope(f::Function, z, z1, L)
    h = 1e-4 * L
    if z - 2h < z1
        return (-25f(z) + 48f(z + h) - 36f(z + 2h) + 16f(z + 3h) - 3f(z + 4h)) / (12h)
    elseif z + 2h > z1 + L
        return (25f(z) - 48f(z - h) + 36f(z - 2h) - 16f(z - 3h) + 3f(z - 4h)) / (12h)
    else
        return (f(z - 2h) - 8f(z - h) + 8f(z + h) - f(z + 2h)) / (12h)
    end
end

"""
    imperfection_vector(model)

Nodal vector `d0` of the initial imperfection (u0, u0', v0, v0', ϕ0, ϕ0' at each
node), the Hermite interpolant of the imperfection shapes.
"""
function imperfection_vector(m::BeamColumnModel)
    d0 = zeros(ndofs(m))
    z1, L = m.z[1], span(m)
    imp = m.imperfection
    for (i, z) in enumerate(m.z), (k, f) in enumerate((imp.u0, imp.v0, imp.ϕ0))
        (f isa Float64 && iszero(f)) && continue
        d0[6 * (i - 1) + 2k - 1] = imperfection_value(f, z, z1, L)
        d0[6 * (i - 1) + 2k]     = imperfection_slope(f, z, z1, L)
    end
    return d0
end

# ------------------------------------------------------------------
# Constraints
# ------------------------------------------------------------------

"""
    constrained_dofs(model)

Global indices of the constrained (zero) degrees of freedom from the end
conditions and the supports.
"""
function constrained_dofs(m::BeamColumnModel)
    dofs = Int[]
    bcdofs(bc) = bc === :fixed ? (1, 2, 3, 4, 5, 6) : bc === :pinned ? (1, 3, 5) : ()
    append!(dofs, bcdofs(m.bc_left))
    append!(dofs, 6 * (nnodes(m) - 1) .+ collect(bcdofs(m.bc_right)))
    for s in m.supports
        base = 6 * (node_at(m, s.z) - 1)
        flags = (s.u, s.u′, s.v, s.v′, s.ϕ, s.ϕ′)
        for k in 1:6
            flags[k] && push!(dofs, base + k)
        end
    end
    return sort(unique(dofs))
end

free_dofs(m::BeamColumnModel) = setdiff(1:ndofs(m), constrained_dofs(m))

# ------------------------------------------------------------------
# Solution
# ------------------------------------------------------------------

"""
    BeamColumnSolution

Result of [`solve_beam_column`](@ref). `d` holds the additional nodal DOFs
(u, u', v, v', ϕ, ϕ') at each node and `d0` the imperfection vector. `stable` is
true when the tangent stiffness on the free DOFs is positive definite, i.e. the
axial load and transverse loads are below the critical level. Evaluate fields
anywhere along the member with [`deflections`](@ref), [`twist`](@ref),
[`lateral_deflection`](@ref), [`vertical_deflection`](@ref),
[`initial_deflections`](@ref) and [`total_deflections`](@ref).
"""
struct BeamColumnSolution
    model::BeamColumnModel
    d::Vector{Float64}
    d0::Vector{Float64}
    residual_norm::Float64
    geometric::Bool
    stable::Bool
    moments::Union{Nothing, Vector{PrebucklingMoments}}   # coupling moments used in Kg, if any
end

"Factor the free-DOF stiffness (sparse Cholesky, sparse LU if indefinite); returns (factorization, stable)."
function factor_free(Kf)
    C = cholesky(Symmetric(Kf); check = false)
    stable = issuccess(C)
    if !stable
        C = lu(Kf; check = false)
        issuccess(C) ||
            error("stiffness matrix K = Ke + Kg is singular; the loads may be at a " *
                  "critical level, or the supports do not prevent rigid-body motion")
    end
    return C, stable
end

"""
    prebuckling_moments(model)

First-order (pre-buckling) bending moments of the transverse loads that drive the
moment-coupling terms, or `nothing` when `model.moment_coupling` is false. They come
from one linear solve with the elastic stiffness alone (no axial force, no
imperfection), so they are exact for statically indeterminate members (interior
supports, fixed ends) and include the bracing reactions. Returns one
[`PrebucklingMoments`](@ref) per element. The moments are proportional to the
transverse loads, so the coupling part of Kg is proportional to q.
"""
function prebuckling_moments(m::BeamColumnModel)
    m.moment_coupling || return nothing
    (all(iszero, m.qx) && all(iszero, m.qy)) &&
        return [PrebucklingMoments(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in 1:nelements(m)]
    K = sparse_elastic_stiffness(m)
    F = load_vector(m)
    free = free_dofs(m)
    C, _ = factor_free(K[free, free])
    d = zeros(ndofs(m))
    d[free] = C \ F[free]
    return map(1:nelements(m)) do e
        de = d[element_dofs(e)]
        g = elastic_element_stiffness(m, e) * de - element_load_vector(m, e)
        s = m.sections[e]; r = m.restraints[e]
        eu, ev = lever_arms(s, r.hx, r.hy)
        PrebucklingMoments(-g[4], g[10], -g[2], g[8],
                           m.qx[e] - r.kx * (de[1] + eu * de[5]), m.qx[e + 1] - r.kx * (de[7] + eu * de[11]),
                           m.qy[e] - r.ky * (de[3] + ev * de[5]), m.qy[e + 1] - r.ky * (de[9] + ev * de[11]))
    end
end

"""
    solve_beam_column(model; geometric=true, warn=true)

Assemble K = Ke + Kg(P, q) and F = F(q) - Kg d0 and solve the linear equilibrium
system K d = F for the free DOFs. Kg depends on the loads but not on the
displacements, so no iteration is needed. K is assembled in sparse storage
([`sparse_elastic_stiffness`](@ref), [`sparse_geometric_stiffness`](@ref)), so
memory and run time grow linearly with the mesh, and is factored by sparse
Cholesky when it is positive definite (stable) and by sparse LU otherwise; an error is thrown if it is
singular (loads exactly at a critical level, or a mechanism).
`residual_norm` is the norm of K d - F equilibrated row-by-row by the stiffness
diagonal (dimensionless). With `geometric=false` the axial-force and load-position
(geometric) stiffness is dropped, which also removes the imperfection loading, and
the first-order elastic response is returned.

A warning is issued (unless `warn=false`) when the tangent stiffness is not
positive definite, i.e. the loads exceed a critical (buckling) level and the
computed equilibrium state is unstable.
"""
function solve_beam_column(m::BeamColumnModel; geometric::Bool = true, warn::Bool = true)
    K = sparse_elastic_stiffness(m)
    d0 = imperfection_vector(m)
    F = load_vector(m)
    moments = geometric ? prebuckling_moments(m) : nothing
    if geometric
        Kg = sparse_geometric_stiffness(m; moments)
        K += Kg
        F .-= Kg * d0
    end

    free = free_dofs(m)
    Kf = K[free, free]
    Ff = F[free]

    C, stable = factor_free(Kf)
    df = C \ Ff

    warn && !stable &&
        @warn "tangent stiffness is not positive definite: the loads exceed a critical " *
              "(buckling) level and the equilibrium state is unstable"

    scale = max.(abs.(diag(Kf)), eps())
    d = zeros(ndofs(m))
    d[free] = df
    return BeamColumnSolution(m, d, d0, norm((Kf * df - Ff) ./ scale), geometric, stable, moments)
end

# ------------------------------------------------------------------
# Field evaluation
# ------------------------------------------------------------------

function eval_field(m::BeamColumnModel, d::AbstractVector, z::Real, field::NTuple{4, Int})
    (m.z[1] <= z <= m.z[end]) || throw(DomainError(z, "z must be within the member"))
    e = clamp(searchsortedlast(m.z, z), 1, nelements(m))
    Le = elength(m, e)
    ξ = (z - m.z[e]) / Le
    N = hermite(ξ, Le)
    dofs = element_dofs(e)
    return sum(N[i] * d[dofs[field[i]]] for i in 1:4)
end

_field(sol::BeamColumnSolution, z, f, total) =
    eval_field(sol.model, sol.d, z, f) + (total ? eval_field(sol.model, sol.d0, z, f) : 0.0)

"Additional lateral deflection u(z) of the shear center (+x); `total=true` adds u0."
lateral_deflection(sol::BeamColumnSolution, z::Real; total::Bool = false) =
    _field(sol, z, UDOF, total)

"Additional vertical deflection v(z) of the shear center (+y, downward); `total=true` adds v0."
vertical_deflection(sol::BeamColumnSolution, z::Real; total::Bool = false) =
    _field(sol, z, VDOF, total)

"Additional twist ϕ(z), positive clockwise; `total=true` adds ϕ0."
twist(sol::BeamColumnSolution, z::Real; total::Bool = false) = _field(sol, z, ΦDOF, total)

"""
    deflections(sol, z; total=false)

Named tuple `(u, v, ϕ)` of the additional (or, with `total=true`, total)
shear-center deflections and twist at `z`.
"""
deflections(sol::BeamColumnSolution, z::Real; total::Bool = false) =
    (u = lateral_deflection(sol, z; total), v = vertical_deflection(sol, z; total),
     ϕ = twist(sol, z; total))

"Named tuple `(u0, v0, ϕ0)` of the initial imperfection at `z` (Hermite interpolant)."
initial_deflections(m::BeamColumnModel, z::Real) = initial_deflections(m, imperfection_vector(m), z)
initial_deflections(sol::BeamColumnSolution, z::Real) = initial_deflections(sol.model, sol.d0, z)
initial_deflections(m::BeamColumnModel, d0::AbstractVector, z::Real) =
    (u0 = eval_field(m, d0, z, UDOF), v0 = eval_field(m, d0, z, VDOF), ϕ0 = eval_field(m, d0, z, ΦDOF))

"Named tuple `(u, v, ϕ)` of the total deflections u0 + u, v0 + v, ϕ0 + ϕ at `z`."
total_deflections(sol::BeamColumnSolution, z::Real) = deflections(sol, z; total = true)

# ------------------------------------------------------------------
# Internal forces at element ends, brace forces
# ------------------------------------------------------------------

"""
    element_end_forces(sol::BeamColumnSolution)

Internal forces, moments, torsion and bimoment at both ends of every element,
recovered by the matrix equilibrium method. The nodal actions on element `e` are

    g = ke dₑ + kg (dₑ + d0ₑ) - fₑ

(elastic stiffness on the additional deformation, geometric stiffness on the total
deformation) and the section actions are `-g[1:6]` at end 1 and `+g[7:12]` at end 2,
so both ends report the force system on the cross-section face with outward normal
+z. Returns one named tuple `(element, end1, end2)` per element, each end a named
tuple `(z, Vx, My, Vy, Mx, T, B)` conjugate to the DOFs (u, u', v, v', ϕ, ϕ'):

  * `Vx`, `Vy` — transverse forces along +x and +y; with an axial load they include
    the transverse component of P (second-order shear), so support reactions are
    recovered directly
  * `My = E (Iy u'' + Ixy v'')`, `Mx = E (Ix v'' + Ixy u'')` — bending moments
    conjugate to u' and v' (second-order, i.e. including the P-δ amplification;
    `Mx(L/2) = -q L²/8` for a first-order pinned-pinned span under gravity)
  * `T` — total torque (St. Venant + warping + axial-force contribution), positive
    clockwise
  * `B = ECw ϕ''` — bimoment, zero at ends free to warp

Discrete braces and supports act at nodes, so the actions jump across a braced or
supported node by the brace force or reaction. Moments are about the centroidal
x-y axes (not necessarily principal).

With moment coupling (`model.moment_coupling`), `axes` selects the reference axes:

  * `:fixed` (default) — the nodal actions use the full kg, including the coupling
    blocks, and are in equilibrium about the fixed (undeformed) x, y, z axes; values at
    coincident element ends agree and statics checks hold exactly. For a member with
    no lateral load, My about the fixed axes stays ≈ 0 even though the section bends
    laterally.
  * `:section` — the coupling blocks are left out, so My ≈ E (Iy u'' + Ixy v'') plus
    the P-δ terms, and T is the resultant about the twisted section's own axes (e.g. a
    weak-axis moment My ≈ Mx ϕ from the rotated strong-axis moment). Use these for
    stress and capacity checks. They are not exactly continuous between elements; the
    jump vanishes with mesh refinement.

Without coupling (or `geometric=false`) both options give the same values.
"""
function element_end_forces(sol::BeamColumnSolution; axes::Symbol = :fixed)
    axes in (:fixed, :section) ||
        throw(ArgumentError("axes must be :fixed or :section, got :$axes"))
    m = sol.model
    moments = axes === :fixed ? sol.moments : nothing
    pack(f, z) = (z = z, Vx = f[1], My = f[2], Vy = f[3], Mx = f[4], T = f[5], B = f[6])
    return map(1:nelements(m)) do e
        dofs = element_dofs(e)
        de, d0e = sol.d[dofs], sol.d0[dofs]
        g = elastic_element_stiffness(m, e) * de - element_load_vector(m, e)
        sol.geometric && (g .+= geometric_element_stiffness(m, e, moments) * (de .+ d0e))
        (element = e, end1 = pack(-g[1:6], m.z[e]), end2 = pack(g[7:12], m.z[e + 1]))
    end
end

"""
    brace_forces(sol::BeamColumnSolution)

Force and moment in each discrete brace: `(z, Fx, Fy, M)` with `Fx = kx u_N`,
`Fy = ky v_N` and `M = kϕ ϕ`, where u_N, v_N are the additional displacements of
the brace point (the paper's brace force kx u₁(αL) and moment kϕ ϕ₁(αL)).
"""
function brace_forces(sol::BeamColumnSolution)
    m = sol.model
    return map(m.braces) do b
        i = node_at(m, b.z)
        s = m.sections[min(i, nelements(m))]
        eu, ev = lever_arms(s, b.hx, b.hy)
        u, v, ϕ = sol.d[6 * (i - 1) + 1], sol.d[6 * (i - 1) + 3], sol.d[6 * (i - 1) + 5]
        (z = b.z, Fx = b.kx * (u + eu * ϕ), Fy = b.ky * (v + ev * ϕ), M = b.kϕ * ϕ)
    end
end

# ------------------------------------------------------------------
# Critical (buckling) loads
# ------------------------------------------------------------------

"""
    critical_load_factors(model; nmodes=1, load_stiffness=true)

Load factors λ such that the axial force distribution λ P(z) makes the perfect
member (imperfections ignored) buckle, from the generalized eigenvalue problem

    [Ke + Kq] d = λ (-KP) d

where KP is the axial geometric stiffness of the reference distribution `P` and Kq
the geometric stiffness of the transverse loads at their given magnitude (load
position and, when `model.moment_coupling` is set, moment coupling; omitted with
`load_stiffness=false`). Returns `(factors, modes)`: the `nmodes` smallest positive
factors in ascending order and the corresponding mode shapes (full DOF vectors,
scaled to unit maximum absolute value). For a pinned-pinned doubly symmetric
column with `P = 1` the first factor is min(π²EIy/L², π²EIx/L², (A/Io)(GJ + π²ECw/L²)).
"""
function critical_load_factors(m::BeamColumnModel; nmodes::Int = 1, load_stiffness::Bool = true)
    any(!iszero, m.P) || throw(ArgumentError("P is zero everywhere: no reference axial load"))
    free = free_dofs(m)
    A = elastic_stiffness(m)
    load_stiffness && (A .+= load_geometric_stiffness(m) .+ moment_coupling_stiffness(m))
    Af = Symmetric(A[free, free])
    Bf = Symmetric(-axial_geometric_stiffness(m)[free, free])

    ch = cholesky(Af; check = false)
    if issuccess(ch)
        # A d = λ B d  ⇔  (L⁻¹ B L⁻ᵀ) y = μ y with y = Lᵀ d, μ = 1/λ
        M = Symmetric(ch.L \ (Matrix(Bf) / ch.U))
        ev = eigen(M)
        μ = ev.values
        idx = sortperm(μ; rev = true)
        idx = filter(i -> μ[i] > 0, idx)[1:min(nmodes, count(>(0), μ))]
        factors = 1 ./ μ[idx]
        vecs = [ch.U \ ev.vectors[:, i] for i in idx]
    else
        ev = eigen(Matrix(Af), Matrix(Bf))
        λ = ev.values
        idx = [i for i in eachindex(λ) if isfinite(λ[i]) && real(λ[i]) > 0 &&
               abs(imag(λ[i])) <= 1e-8 * abs(λ[i])]
        idx = idx[sortperm(real.(λ[idx]))][1:min(nmodes, length(idx))]
        factors = real.(λ[idx])
        vecs = [real.(ev.vectors[:, i]) for i in idx]
    end
    isempty(factors) && error("no positive critical load factor found")
    modes = map(vecs) do y
        d = zeros(ndofs(m))
        d[free] = y ./ maximum(abs, y)
        d
    end
    return (factors = factors, modes = modes)
end

"""
    critical_transverse_load_factor(model)

Smallest positive factor λ on the transverse loads `qx`, `qy` at which the member,
carrying the axial force `P` as given, buckles: `[Ke + KP] d = λ (-Kq) d`, where Kq is
the geometric stiffness of the transverse loads (load position and moment coupling,
both proportional to q). With `P = 0` and the moment coupling on this is the elastic
lateral-torsional buckling load of a beam; Ke + KP must be positive definite on the
free DOFs (P below its own critical level). Returns `Inf` when no positive factor
exists (e.g. uplift with the load-height term only).
"""
function critical_transverse_load_factor(m::BeamColumnModel)
    (any(!iszero, m.qx) || any(!iszero, m.qy)) ||
        throw(ArgumentError("qx and qy are zero everywhere: no reference transverse load"))
    free = free_dofs(m)
    A = elastic_stiffness(m) .+ axial_geometric_stiffness(m)
    Af = Symmetric(A[free, free])
    Kq = (load_geometric_stiffness(m) .+ moment_coupling_stiffness(m))[free, free]
    ch = cholesky(Af; check = false)
    issuccess(ch) || error("Ke + KP is not positive definite: P is at or beyond its critical level")
    M = Symmetric(ch.L \ (Matrix(-Kq) / ch.U))         # -L⁻¹ Kq L⁻ᵀ
    μ = maximum(eigvals(M))
    return μ > 0 ? 1 / μ : Inf
end

# ------------------------------------------------------------------
# ThinWalledBeamColumn.jl-compatible interface
# ------------------------------------------------------------------

"""
    Inputs

Nodal input arrays of the ThinWalledBeamColumn.jl-style interface, see [`solve`](@ref).
"""
struct Inputs
    z::Vector{Float64}
    A::Vector{Float64}
    Ix::Vector{Float64}
    Iy::Vector{Float64}
    Ixy::Vector{Float64}
    Io::Vector{Float64}
    J::Vector{Float64}
    Cw::Vector{Float64}
    E::Vector{Float64}
    G::Vector{Float64}
    ax::Vector{Float64}
    ay::Vector{Float64}
    kx::Vector{Float64}
    ky::Vector{Float64}
    kϕ::Vector{Float64}
    hx::Vector{Float64}
    hy::Vector{Float64}
    xo::Vector{Float64}
    yo::Vector{Float64}
    qx::Vector{Float64}
    qy::Vector{Float64}
    P::Vector{Float64}
    end_boundary_conditions::Vector{String}
    supports::Vector{Tuple{Float64, String, String, String}}
end

"Nodal deformations `u`, `v`, `ϕ` (additional to any imperfection)."
struct Outputs
    u::Vector{Float64}
    v::Vector{Float64}
    ϕ::Vector{Float64}
end

"""
    Model

Result of the ThinWalledBeamColumn.jl-style [`solve`](@ref): `inputs`, `outputs` (nodal
`u`, `v`, `ϕ`), and the underlying finite element `model::BeamColumnModel` and
`solution::BeamColumnSolution`.
"""
struct Model
    inputs::Inputs
    outputs::Outputs
    model::BeamColumnModel
    solution::BeamColumnSolution
end

"""
    solve(z, A, Ix, Iy, Io, J, Cw, E, G, ax, ay, kx, ky, kϕ, hx, hy, qx, qy, P,
          end_boundary_conditions, supports; xo=0, yo=0, Ixy=0, imperfection=Imperfection(),
          moment_coupling=true, warn=false)

Drop-in replacement for `ThinWalledBeamColumn.solve`. All property, restraint and load
arguments are arrays of nodal values at the coordinates `z` (element properties are
the average of the two end values; loads and P vary linearly inside each element).

  * `end_boundary_conditions = [left, right]` with `"simply-supported"` (u'' = v'' = ϕ'' = 0,
    a natural condition of the finite element), `"fixed"` (u' = v' = ϕ' = 0 imposed at the
    end node) or `"free"` (cantilever tip).
  * `supports = [(z, u, v, ϕ), ...]` with `"fixed"` or `"free"` for each of u, v, ϕ; a
    support may be placed at any node, including the ends.
  * `xo`, `yo` — shear center offsets from the centroid (y downward), which add the axial
    load coupling of Plaut & Moen (2020) Eqs. (1)-(3) that ThinWalledBeamColumn.jl omits.
    With the defaults `xo = yo = 0` the two packages solve the same equations.
  * `hx`, `hy` locate the springs from the centroid (ThinWalledBeamColumn.jl effectively
    measured them from the shear center; identical when `xo = yo = 0`).
  * `Ixy` — optional product of inertia coupling the two bending equations.
  * `moment_coupling` — lateral-torsional coupling of the first-order moments with the
    twist (default `true`); `false` recovers the ThinWalledBeamColumn.jl equations.

Returns a [`Model`](@ref) whose `outputs.u`, `outputs.v`, `outputs.ϕ` are the nodal
deformations, as in ThinWalledBeamColumn.jl.
"""
function solve(z, A, Ix, Iy, Io, J, Cw, E, G, ax, ay, kx, ky, kϕ, hx, hy, qx, qy, P,
               end_boundary_conditions, supports;
               xo = zeros(length(z)), yo = zeros(length(z)), Ixy = zeros(length(z)),
               imperfection::Imperfection = Imperfection(), moment_coupling::Bool = true,
               warn::Bool = false)
    zz = Float64.(collect(z))
    n = length(zz)
    arr(x) = (length(x) == n || throw(ArgumentError("expected $n nodal values, got $(length(x))"));
              Float64.(collect(x)))
    A, Ix, Iy, Ixy, Io, J, Cw, E, G = arr.((A, Ix, Iy, Ixy, Io, J, Cw, E, G))
    ax, ay, kx, ky, kϕ, hx, hy, xo, yo = arr.((ax, ay, kx, ky, kϕ, hx, hy, xo, yo))
    qx, qy, P = arr.((qx, qy, P))
    length(end_boundary_conditions) == 2 ||
        throw(ArgumentError("end_boundary_conditions must have two entries"))
    for bc in end_boundary_conditions
        bc in ("simply-supported", "fixed", "free") ||
            throw(ArgumentError("end boundary condition must be \"simply-supported\", \"fixed\" or \"free\", got \"$bc\""))
    end

    ne = n - 1
    mid(x) = [(x[i] + x[i + 1]) / 2 for i in 1:ne]
    sections = [Section(A = A_, Ix = Ix_, Iy = Iy_, Ixy = Ixy_, Io = Io_, J = J_, Cw = Cw_,
                        xo = xo_, yo = yo_, ax = ax_, ay = ay_)
                for (A_, Ix_, Iy_, Ixy_, Io_, J_, Cw_, xo_, yo_, ax_, ay_) in
                    zip(mid(A), mid(Ix), mid(Iy), mid(Ixy), mid(Io), mid(J), mid(Cw),
                        mid(xo), mid(yo), mid(ax), mid(ay))]
    materials = [Material(E = E_, G = G_) for (E_, G_) in zip(mid(E), mid(G))]
    restraints = [Restraint(kx = kx_, ky = ky_, kϕ = kϕ_, hx = hx_, hy = hy_)
                  for (kx_, ky_, kϕ_, hx_, hy_) in zip(mid(kx), mid(ky), mid(kϕ), mid(hx), mid(hy))]

    fixed(flag) = flag == "fixed" ? true : flag == "free" ? false :
                  throw(ArgumentError("support condition must be \"fixed\" or \"free\", got \"$flag\""))
    fe_supports = [Support(Float64(s[1]); u = fixed(s[2]), v = fixed(s[3]), ϕ = fixed(s[4]))
                   for s in supports]
    # "fixed" ends restrain the rotations and warping; translations come from `supports`
    for (bc, zend) in zip(end_boundary_conditions, (zz[1], zz[end]))
        bc == "fixed" && push!(fe_supports, Support(zend; u = false, v = false, ϕ = false,
                                                    u′ = true, v′ = true, ϕ′ = true))
    end

    model = BeamColumnModel(z = zz, section = sections, material = materials,
                            restraint = restraints, supports = fe_supports,
                            imperfection = imperfection, qx = qx, qy = qy, P = P,
                            bc_left = :free, bc_right = :free, moment_coupling = moment_coupling)
    sol = solve_beam_column(model; warn = warn)

    inputs = Inputs(zz, A, Ix, Iy, Ixy, Io, J, Cw, E, G, ax, ay, kx, ky, kϕ, hx, hy, xo, yo,
                    qx, qy, P, String.(collect(end_boundary_conditions)),
                    [(Float64(s[1]), String(s[2]), String(s[3]), String(s[4])) for s in supports])
    outputs = Outputs(sol.d[1:6:end], sol.d[3:6:end], sol.d[5:6:end])
    return Model(inputs, outputs, model, sol)
end

end # module PlautBeamColumnElement
