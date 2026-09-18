using PlautBeamColumnElement
using Test
using LinearAlgebra

const PBC = PlautBeamColumnElement

# ==================================================================
# Reference data from Plaut & Moen (2020), Thin-Walled Structures 154, 106897
# Units: N, mm (E = 200 kN/mm² = 200e3 N/mm²; 1 kN·mm/rad = 1e3 N·mm/rad)
# ==================================================================

const L = 2438.0
const mat = Material(E = 200e3)                      # G = E/2.6

# Example 1: 362S162-54 lipped C stud (Section 3); shear center at x0 = -32.59 mm
const sec1 = Section(A = 272.0, Ix = 363370.0, Iy = 64100.0, J = 188.0,
                     Cw = 122720891.0, xo = -32.59, yo = 0.0)

# Example 2: W18×35 doubly symmetric I (Section 4)
const sec2 = Section(A = 6650.0, Ix = 2.12e8, Iy = 6.37e6, Io = 2.19e8,
                     J = 2.11e5, Cw = 3.06e11)

# Standard imperfection amplitudes, Eq. (10)
const a1 = L / 1000
const a2 = L / 1000
const a3 = 0.00766
const imp = Imperfection(u0 = a1, v0 = a2, ϕ0 = a3)

# Classical critical loads of a pinned-pinned column
Pe_y(s) = π^2 * mat.E * s.Iy / L^2
Pe_x(s) = π^2 * mat.E * s.Ix / L^2
Pe_ϕ(s) = (s.A / s.Io) * (mat.G * s.J + π^2 * mat.E * s.Cw / L^2)
function Pe_ft(s)                                    # v-ϕ coupled, yo = 0
    Px, Pϕ = Pe_x(s), Pe_ϕ(s)
    β = 1 - s.xo^2 * s.A / s.Io
    return ((Px + Pϕ) - sqrt((Px + Pϕ)^2 - 4β * Px * Pϕ)) / (2β)
end

"""
Exact Fourier sine-series solution of the governing equations for a pinned-pinned
member with uniform properties, uniform continuous restraint, uniform P, uniform
qx, qy and half-sine imperfections (a1, a2, a3). With constant coefficients the
equations decouple mode by mode into a 3×3 system for (uₙ, vₙ, ϕₙ); the uniform
loads expand as Σ (4/nπ) sin(λₙz) over odd n and the imperfection is mode 1 only.
"""
function series_solution(z; s, r = Restraint(), qx = 0.0, qy = 0.0, P = 0.0,
                         a = (0.0, 0.0, 0.0), nmodes = 399)
    (; E, G) = mat
    eu, ev = s.yo - r.hy, r.hx - s.xo
    w = qx * s.ax - qy * s.ay
    u = v = ϕ = 0.0
    for n in 1:nmodes
        λ = n * π / L
        c = isodd(n) ? 4 / (n * π) : 0.0
        (iszero(c) && n != 1) && continue
        A = [E*s.Iy*λ^4 - P*λ^2 + r.kx     E*s.Ixy*λ^4                     r.kx*eu - P*s.yo*λ^2
             E*s.Ixy*λ^4                   E*s.Ix*λ^4 - P*λ^2 + r.ky       r.ky*ev + P*s.xo*λ^2
             r.kx*eu - P*s.yo*λ^2          r.ky*ev + P*s.xo*λ^2            E*s.Cw*λ^4 + G*s.J*λ^2 - P*(s.Io/s.A)*λ^2 + r.kx*eu^2 + r.ky*ev^2 + r.kϕ + w]
        b = c .* [qx, qy, qx * s.ay + qy * s.ax]
        if n == 1
            b .+= [P * λ^2 * (a[1] + s.yo * a[3]),
                   P * λ^2 * (a[2] - s.xo * a[3]),
                   P * λ^2 * (s.yo * a[1] - s.xo * a[2] + (s.Io / s.A) * a[3]) - w * a[3]]
        end
        x = A \ b
        sn = sin(λ * z)
        u += x[1] * sn; v += x[2] * sn; ϕ += x[3] * sn
    end
    return (u = u, v = v, ϕ = ϕ)
end

model1(; nel = 40, kwargs...) = BeamColumnModel(; L, nel, section = sec1, material = mat, kwargs...)
model2(; nel = 40, kwargs...) = BeamColumnModel(; L, nel, section = sec2, material = mat, kwargs...)

function peak_total_twist(sol; npts = 2439)
    zs = range(0, L, npts)
    ϕ = [twist(sol, z; total = true) for z in zs]
    i = argmax(ϕ)
    return (ϕ[i], zs[i])
end

# ==================================================================

@testset "PlautBeamColumnElement" begin

@testset "element kernels" begin
    Le = 1.7
    @test PBC.k_grad_linear(Le, 1.0, 1.0) ≈ PBC.k_grad(Le) atol = 1e-14
    @test PBC.k_found_linear(Le, 1.0, 1.0) ≈ PBC.k_found(Le) atol = 1e-14
    @test PBC.f_linear(Le, 1.0, 1.0) ≈ [Le / 2, Le^2 / 12, Le / 2, -Le^2 / 12]
    # linear weights: ∫ ξ N'ᵀN' + ∫ (1-ξ) N'ᵀN' = ∫ N'ᵀN'
    @test PBC.k_grad_linear(Le, 0.0, 1.0) + PBC.k_grad_linear(Le, 1.0, 0.0) ≈ PBC.k_grad(Le)
    # Hermite derivatives: d/dz of the shape functions by finite differences
    for ξ in (0.1, 0.5, 0.9)
        h = 1e-6
        fd = (collect(PBC.hermite(ξ + h, Le)) .- collect(PBC.hermite(ξ - h, Le))) ./ (2h * Le)
        @test collect(PBC.hermite_d(ξ, Le)) ≈ fd atol = 1e-6
    end
end

@testset "stiffness matrix properties" begin
    r = Restraint(kx = 0.05, ky = 0.01, kϕ = 100.0, hx = 5.0, hy = 225.0)
    m = model2(P = 1000e3, qy = 1.0, qx = 0.2, restraint = r,
               braces = [Brace(L / 2; kx = 5e3, kϕ = 1e6, hy = 225.0)])
    Ke, KP, Kq = elastic_stiffness(m), axial_geometric_stiffness(m), load_geometric_stiffness(m)
    @test Ke ≈ Ke'
    @test KP ≈ KP'
    @test Kq ≈ Kq'
    @test geometric_stiffness(m) ≈ KP + Kq
    free = setdiff(1:size(Ke, 1), constrained_dofs(m))
    @test isposdef(Symmetric(Ke[free, free]))
    # compression softens: KP negative semi-definite (Io/A > 0, and the 3×3
    # coupling matrix [1 0 yo; 0 1 -xo; yo -xo Io/A] is positive definite)
    @test eigmax(Symmetric(KP[free, free])) <= 1e-9 * abs(eigmin(Symmetric(KP[free, free])))
    # tension stiffens
    KPt = axial_geometric_stiffness(model2(P = -1000e3))
    @test eigmin(Symmetric(KPt[free, free])) >= -1e-9 * eigmax(Symmetric(KPt[free, free]))
    @test iszero(axial_geometric_stiffness(model2(P = 0.0)))
    @test iszero(load_geometric_stiffness(model2(qy = 1.0)))          # ax = ay = 0
    # the brace adds exactly its 3×3 kernel at the midspan node
    K0 = elastic_stiffness(model2(restraint = r))
    ΔK = Ke - K0
    i = 6 * 20 .+ [1, 3, 5]
    @test ΔK[i, i] ≈ PBC.spring_kernel(5e3, 0.0, 1e6, -225.0, 0.0)
    ΔK[i, i] .= 0
    @test iszero(ΔK)
end

@testset "first-order bending benchmarks (P = 0)" begin
    q = 1.0
    EIx = mat.E * sec2.Ix
    sol = solve_beam_column(model2(qy = q))
    @test vertical_deflection(sol, L / 2) ≈ 5q * L^4 / (384EIx) rtol = 1e-9
    sol = solve_beam_column(model2(qy = q, bc_left = :fixed, bc_right = :fixed))
    @test vertical_deflection(sol, L / 2) ≈ q * L^4 / (384EIx) rtol = 1e-9
    sol = solve_beam_column(model2(qy = q, bc_right = :fixed))
    @test vertical_deflection(sol, L / 2) ≈ q * L^4 / (192EIx) rtol = 1e-9
    # cantilever: fixed at z = 0, free tip
    sol = solve_beam_column(model2(qy = q, bc_left = :fixed, bc_right = :free))
    @test vertical_deflection(sol, L) ≈ q * L^4 / (8EIx) rtol = 1e-9
    # no P, no imperfection effect, linear in q
    sol2 = solve_beam_column(model2(qy = 2q, imperfection = imp))
    @test vertical_deflection(sol2, L / 2) ≈ 2 * 5q * L^4 / (384EIx) rtol = 1e-9
end

@testset "beam-column: exact secant solution (pinned-pinned, uniform qx + P)" begin
    EI = mat.E * sec2.Iy
    Pe = Pe_y(sec2)
    q = 5.0
    for ratio in (0.2, 0.5, 0.8)
        P = ratio * Pe
        k = sqrt(P / EI)
        u_exact = (q / (P * k^2)) * (sec(k * L / 2) - 1) - q * L^2 / (8P)
        M_exact = (q / k^2) * (sec(k * L / 2) - 1)
        sol = solve_beam_column(model2(P = P, qx = q))
        @test sol.stable
        @test lateral_deflection(sol, L / 2) ≈ u_exact rtol = 1e-6
        f = element_end_forces(sol)
        @test f[20].end2.My ≈ -M_exact rtol = 1e-6            # second-order moment, P-δ included
        @test f[1].end1.Vx ≈ q * L / 2 rtol = 1e-8            # reactions unchanged by P
        @test f[end].end2.Vx ≈ -q * L / 2 rtol = 1e-8
        # tension reduces the deflection below first order
        solt = solve_beam_column(model2(P = -P, qx = q))
        @test lateral_deflection(solt, L / 2) < 5q * L^4 / (384EI)
    end
    # first-order response with geometric = false
    sol = solve_beam_column(model2(P = 0.5Pe, qx = q); geometric = false)
    @test lateral_deflection(sol, L / 2) ≈ 5q * L^4 / (384EI) rtol = 1e-9
end

@testset "critical loads: classical columns" begin
    # pinned-pinned doubly symmetric: weak-axis Euler load, then torsional
    cr = critical_load_factors(model2(P = 1.0); nmodes = 3)
    @test cr.factors[1] ≈ Pe_y(sec2) rtol = 1e-6
    @test cr.factors[1] ≈ 2115e3 rtol = 2e-3                  # Section 4.2: "2115 kN"
    @test cr.factors[2] ≈ Pe_ϕ(sec2) rtol = 1e-6              # torsional mode next (3578 kN)
    @test cr.factors[3] ≈ 4Pe_y(sec2) rtol = 1e-5             # second flexural mode
    # mode shape of mode 1 is a half sine in u
    d = cr.modes[1]
    @test abs(d[6 * 20 + 1]) ≈ 1.0                            # u at midspan, normalized
    @test abs(d[6 * 10 + 1]) ≈ sin(π / 4) rtol = 1e-4
    # fixed-fixed: 4 π² EI / L²
    @test critical_load_factors(model2(P = 1.0, bc_left = :fixed, bc_right = :fixed)).factors[1] ≈
          4Pe_y(sec2) rtol = 1e-5
    # fixed-free (flagpole): π² EI / (4 L²)
    @test critical_load_factors(model2(P = 1.0, bc_left = :fixed, bc_right = :free)).factors[1] ≈
          Pe_y(sec2) / 4 rtol = 1e-6
    # rigid lateral support at midspan: second flexural mode 4 Pe, but the torsional
    # mode (3578 kN) now governs
    @test critical_load_factors(model2(P = 1.0, supports = [Support(L / 2; v = false, ϕ = false)])).factors[1] ≈
          min(4Pe_y(sec2), Pe_ϕ(sec2)) rtol = 1e-5
    @test critical_load_factors(model2(P = 1.0, supports = [Support(L / 2; v = false)])).factors[1] ≈
          4Pe_y(sec2) rtol = 1e-5
    # a very stiff discrete brace at the centroid tends to the rigid support
    @test critical_load_factors(model2(P = 1.0, braces = [Brace(L / 2; kx = 1e12)])).factors[1] ≈
          min(4Pe_y(sec2), Pe_ϕ(sec2)) rtol = 1e-4
    # P is a reference distribution: the factor scales inversely
    @test critical_load_factors(model2(P = 1000.0)).factors[1] ≈ Pe_y(sec2) / 1000 rtol = 1e-6
end

@testset "critical loads: flexural-torsional buckling of the C stud (Example 1)" begin
    # singly symmetric section with xo ≠ 0: v and ϕ couple, Eqs. (2)-(3)
    cr = critical_load_factors(model1(P = 1.0); nmodes = 3)
    @test cr.factors[1] ≈ Pe_ft(sec1) rtol = 1e-6            # 19.46 kN, the coupled mode
    @test cr.factors[2] ≈ Pe_y(sec1) rtol = 1e-6             # 21.29 kN, weak-axis flexure
    @test Pe_ft(sec1) < min(Pe_ϕ(sec1), Pe_x(sec1))          # coupling lowers the load
    # without the shear-center coupling the coupled mode is lost
    sec1_uncoupled = Section(A = sec1.A, Ix = sec1.Ix, Iy = sec1.Iy, J = sec1.J,
                             Cw = sec1.Cw, Io = sec1.Io)
    m0 = BeamColumnModel(L = L, nel = 40, section = sec1_uncoupled, material = mat, P = 1.0)
    @test critical_load_factors(m0).factors[1] ≈ Pe_ϕ(sec1) rtol = 1e-6
    # a rigid midspan torsional brace raises the first load above the pinned-pinned FT load
    crb = critical_load_factors(model1(P = 1.0, braces = [Brace(L / 2; kϕ = 1e12)]))
    @test crb.factors[1] > Pe_ft(sec1)
    @test crb.factors[1] ≈ Pe_y(sec1) rtol = 1e-4              # weak-axis flexure governs
end

@testset "pinned-pinned members vs exact series (Examples 1 and 2)" begin
    cases = [
        # (section, restraint, qx, qy, P, imperfection amplitudes)
        (sec2, Restraint(),                                        0.0, 0.0, 1000e3, (a1, a2, a3)),
        (sec2, Restraint(kx = 0.05, hy = 225.0),                   0.0, 0.0, 1000e3, (a1, a2, a3)),
        (sec2, Restraint(kx = 0.05, ky = 0.02, kϕ = 200.0, hx = 30.0, hy = 225.0),
                                                                   0.3, 1.0, 800e3,  (a1, -a2, a3)),
        (sec2, Restraint(kϕ = 50.0),                               0.0, 0.0, 1500e3, (0.0, 0.0, a3)),
        (sec1, Restraint(),                                        0.0, 0.0, 10e3,   (a1, a2, a3)),
        (sec1, Restraint(kϕ = 0.1),                                0.0, 0.0, 15e3,   (a1, a2, a3)),
        (sec1, Restraint(kx = 0.02, kϕ = 0.05, hy = -46.05),       0.0, 0.0, 12e3,   (a1, -a2, -a3)),
        (sec1, Restraint(kx = 0.02, kϕ = 0.05, hy = -46.05),       0.0, 0.0, 12e3,   (0.0, 0.0, 0.0)),
    ]
    for (s, r, qx, qy, P, a) in cases
        m = BeamColumnModel(L = L, nel = 40, section = s, material = mat, restraint = r,
                            qx = qx, qy = qy, P = P, imperfection = Imperfection(u0 = a[1], v0 = a[2], ϕ0 = a[3]))
        sol = solve_beam_column(m)
        for z in (0.25L, 0.42L, 0.5L)
            ref = series_solution(z; s, r, qx, qy, P, a)
            scale = max(abs(ref.u), abs(ref.v), 1e-3 * a1)
            @test lateral_deflection(sol, z) ≈ ref.u rtol = 1e-4 atol = 1e-6 * scale
            @test vertical_deflection(sol, z) ≈ ref.v rtol = 1e-4 atol = 1e-6 * scale
            @test twist(sol, z) ≈ ref.ϕ rtol = 1e-4 atol = 1e-9
        end
    end
    # transverse loads on a section with load offsets ax, ay (JCSR-type terms)
    s = Section(A = sec1.A, Ix = sec1.Ix, Iy = sec1.Iy, J = sec1.J, Cw = sec1.Cw, xo = sec1.xo,
                ax = 20.65, ay = 46.05)
    for (qx, qy, P) in ((0.0, 0.05, 5e3), (0.02, -0.05, 8e3), (0.0, 0.08, 0.0))
        r = Restraint(kx = 0.01, kϕ = 0.05, hy = -46.05)
        m = BeamColumnModel(L = L, nel = 40, section = s, material = mat, restraint = r,
                            qx = qx, qy = qy, P = P, imperfection = imp)
        sol = solve_beam_column(m)
        for z in (0.25L, 0.5L)
            ref = series_solution(z; s, r, qx, qy, P, a = (a1, a2, a3))
            @test lateral_deflection(sol, z) ≈ ref.u rtol = 1e-4
            @test vertical_deflection(sol, z) ≈ ref.v rtol = 1e-4
            @test twist(sol, z) ≈ ref.ϕ rtol = 1e-4
        end
    end
end

@testset "imperfection amplification (doubly symmetric, unbraced)" begin
    Pe, Pϕ = Pe_y(sec2), Pe_ϕ(sec2)
    for ratio in (0.25, 0.5, 0.75)
        sol = solve_beam_column(model2(P = ratio * Pe, imperfection = Imperfection(u0 = 2.0)))
        @test lateral_deflection(sol, L / 2; total = true) ≈ 2.0 / (1 - ratio) rtol = 1e-6
        @test lateral_deflection(sol, L / 2) ≈ 2.0 * ratio / (1 - ratio) rtol = 1e-6
        @test vertical_deflection(sol, L / 2) ≈ 0.0 atol = 1e-12
        @test twist(sol, L / 2) ≈ 0.0 atol = 1e-15
        solϕ = solve_beam_column(model2(P = ratio * Pϕ, imperfection = Imperfection(ϕ0 = a3)))
        @test twist(solϕ, L / 2; total = true) ≈ a3 / (1 - ratio) rtol = 1e-6
    end
    # Fig. 12 at kx = 0: total twist a3/(1 - P/Pϕ) for P = 500, 1000, 1500 kN
    for P in (500e3, 1000e3, 1500e3)
        sol = solve_beam_column(model2(P = P, imperfection = imp))
        @test twist(sol, L / 2; total = true) ≈ a3 / (1 - P / Pϕ) rtol = 1e-6
    end
    # function-valued imperfection equals the amplitude form
    f = Imperfection(u0 = z -> 2.0 * sin(π * z / L), ϕ0 = z -> a3 * sin(π * z / L))
    sa = solve_beam_column(model2(P = 0.5Pe, imperfection = Imperfection(u0 = 2.0, ϕ0 = a3)))
    sf = solve_beam_column(model2(P = 0.5Pe, imperfection = f))
    @test sf.d ≈ sa.d rtol = 1e-8
    @test initial_deflections(sf, 0.3L).u0 ≈ 2.0 * sin(0.3π) rtol = 1e-12
    @test initial_deflections(model2(imperfection = f), 0.3L).ϕ0 ≈ a3 * sin(0.3π) rtol = 1e-12
    @test total_deflections(sf, 0.3L).u ≈ lateral_deflection(sf, 0.3L; total = true)
    # a full-sine (antisymmetric) imperfection excites mode 2 only: amplification 1/(1 - P/4Pe)
    s2 = solve_beam_column(model2(P = 0.5Pe, imperfection = Imperfection(u0 = z -> 2.0 * sin(2π * z / L))))
    @test lateral_deflection(s2, L / 4; total = true) ≈ 2.0 / (1 - 0.5 / 4) rtol = 1e-5
    # imperfections have no effect without the geometric stiffness
    @test iszero(solve_beam_column(model2(P = 0.5Pe, imperfection = imp); geometric = false).d)
end

@testset "Example 1: C stud with discrete torsional brace (Section 3)" begin
    # α = 1/2, P = 10 kN: kϕ = 230.7 kN·mm/rad makes the additional twist equal
    # to the initial twist a3 (Winter's criterion, Section 3.2)
    sol = solve_beam_column(model1(P = 10e3, imperfection = imp, braces = [Brace(L / 2; kϕ = 230.7e3)]))
    @test twist(sol, L / 2) ≈ a3 rtol = 0.03
    @test twist(sol, L / 2; total = true) ≈ 2a3 rtol = 0.03
    @test lateral_deflection(sol, L / 2) ≈ a1 * (10e3 / Pe_y(sec1)) / (1 - 10e3 / Pe_y(sec1)) rtol = 1e-5  # u uncoupled (yo = 0)
    @test brace_forces(sol)[1].M ≈ 230.7e3 * twist(sol, L / 2)
    # Fig. 5 middle curve: at P = 5 kN the required kϕ is lower, at 15 kN higher
    g(P, kϕ) = twist(solve_beam_column(model1(P = P, imperfection = imp, braces = [Brace(L / 2; kϕ = kϕ)])), L / 2)
    @test g(5e3, 230.7e3) < a3 < g(15e3, 230.7e3)
    # Fig. 3: total midheight twist decreases with kϕ and increases with P
    @test g(10e3, 50e3) > g(10e3, 200e3) > g(10e3, 600e3) > 0
    @test g(5e3, 100e3) < g(10e3, 100e3) < g(15e3, 100e3)
    # Fig. 4 (kϕ = 50, P = 10 kN): a2 = +L/1000 gives the largest twist, a2 = -L/1000 a negative one
    h(a2) = twist(solve_beam_column(model1(P = 10e3, imperfection = Imperfection(u0 = a1, v0 = a2, ϕ0 = a3),
                                          braces = [Brace(L / 2; kϕ = 50e3)])), L / 2; total = true)
    @test h(a2) > h(0.0) > h(-a2)
    @test h(-a2) < 0
    # Fig. 6: for a given kϕ the midheight twist grows as the brace moves toward the end
    t(α, kϕ) = twist(solve_beam_column(model1(P = 10e3, imperfection = imp, braces = [Brace(α * L; kϕ = kϕ)])), L / 2; total = true)
    @test t(0.2, 300e3) > t(0.3, 300e3) > t(0.5, 300e3)
    # Fig. 9: α = 0.2, P = 10 kN, kϕ = 500 kN·mm/rad: peak total twist 0.021 rad at z = 1337 mm
    sol = solve_beam_column(model1(P = 10e3, imperfection = imp, braces = [Brace(0.2L; kϕ = 500e3)]))
    ϕmax, zmax = peak_total_twist(sol)
    @test ϕmax ≈ 0.021 rtol = 0.03
    @test abs(zmax - 1337.0) < 25.0
    @test twist(sol, 0.0; total = true) ≈ 0.0 atol = 1e-12
end

@testset "Example 2: I column with offset lateral brace (Section 4)" begin
    # Fig. 11: FTB load 2115 kN unbraced, rising to ≈ 2670 kN at kx = 20 kN/mm
    br(kx) = [Brace(L / 2; kx = kx, hy = 225.0)]
    @test critical_load_factors(model2(P = 1.0)).factors[1] ≈ 2115e3 rtol = 2e-3
    P20 = critical_load_factors(model2(P = 1.0, braces = br(20e3))).factors[1]
    @test 2600e3 < P20 < 2700e3
    P5 = critical_load_factors(model2(P = 1.0, braces = br(5e3))).factors[1]
    @test 2115e3 < P5 < P20
    # Fig. 12 (α = 1/2, kx = 10 kN/mm): total midheight twist ≈ 0.0097, 0.0130, 0.0192 rad
    for (P, ϕref) in ((500e3, 0.0097), (1000e3, 0.0130), (1500e3, 0.0192))
        sol = solve_beam_column(model2(P = P, imperfection = imp, braces = br(10e3)))
        @test twist(sol, L / 2; total = true) ≈ ϕref rtol = 0.03
        # brace force is kx u_N with u_N = u - hy ϕ (Section 4.1)
        bf = brace_forces(sol)[1]
        @test bf.Fx ≈ 10e3 * (lateral_deflection(sol, L / 2) - 225.0 * twist(sol, L / 2))
    end
    # Fig. 12: unlike the torsional brace, the offset lateral brace increases the twist
    tw(kx, P) = twist(solve_beam_column(model2(P = P, imperfection = imp, braces = br(kx))), L / 2; total = true)
    @test tw(0.0, 1000e3) < tw(2e3, 1000e3) < tw(10e3, 1000e3)
    # Fig. 13 (kx = 1 kN/mm, P = 1000 kN): a1 = +L/1000 largest twist, all positive
    t13(a) = twist(solve_beam_column(model2(P = 1000e3, imperfection = Imperfection(u0 = a, v0 = a2, ϕ0 = a3),
                                            braces = br(1e3))), L / 2; total = true)
    @test t13(a1) > t13(0.0) > t13(-a1) > 0
    # Fig. 14 (P = 1000 kN, kx = 10 kN/mm): twist decreases as the brace moves toward the end
    t14(α) = twist(solve_beam_column(model2(P = 1000e3, imperfection = imp,
                                            braces = [Brace(α * L; kx = 10e3, hy = 225.0)])), L / 2; total = true)
    @test t14(0.5) > t14(0.3) > t14(0.2)
    # Fig. 16: α = 0.2, P = 1500 kN, kx = 100 kN/mm: peak total twist 0.019 rad at z = 1185 mm
    sol = solve_beam_column(model2(P = 1500e3, imperfection = imp, braces = [Brace(0.2L; kx = 100e3, hy = 225.0)]))
    ϕmax, zmax = peak_total_twist(sol)
    @test ϕmax ≈ 0.019 rtol = 0.03
    @test abs(zmax - 1185.0) < 25.0
end

@testset "continuous restraint: JCSR beam terms and rigid-body twist" begin
    # A lateral spring on the top flange of a section with xo = yo = 0 is the JCSR
    # term kx (u + ay ϕ) with hy = -ay; the ϕϕ term is kx ay² + kϕ.
    s = Section(A = 1.0, Ix = 4.287e6, Iy = 290500.0, J = 424.56, Cw = 3.1956e9, Io = 1.0,
                ax = 57.67, ay = 114.3)
    r = Restraint(kx = 0.1, kϕ = 1000.0, hy = -114.3)
    m = BeamColumnModel(L = 7620.0, nel = 40, section = s, material = mat, restraint = r, qy = 1.0)
    sol = solve_beam_column(m)
    # values from the JCSR paper, Fig. 7 / PlautBeamFiniteElement: ϕ(L/2) ≈ 0.072, u(L/2) ≈ -8.5 mm
    @test 0.06 < twist(sol, 3810.0) < 0.09
    @test -10.0 < lateral_deflection(sol, 3810.0) < -7.0
    # pinned in u, v but free to twist at both ends, uniform torque against a
    # continuous rotational spring: rigid-body twist ϕ = qy ax / kϕ
    s2 = Section(A = sec2.A, Ix = sec2.Ix, Iy = sec2.Iy, J = sec2.J, Cw = sec2.Cw, Io = sec2.Io, ax = 50.0)
    m = BeamColumnModel(L = L, nel = 20, section = s2, material = mat, restraint = Restraint(kϕ = 2000.0),
                        qy = 1.0, bc_left = :free, bc_right = :free,
                        supports = [Support(0.0; ϕ = false), Support(L; ϕ = false)])
    sol = solve_beam_column(m)
    for z in (0.0, 0.3L, L)
        @test twist(sol, z) ≈ 1.0 * 50.0 / 2000.0 rtol = 1e-9
    end
end

@testset "varying axial force" begin
    # scalar, 2-tuple, vector and function inputs agree
    m1 = model2(P = (1000e3, 0.0), qx = 1.0)
    m2 = model2(P = [1000e3 * (1 - z / L) for z in range(0, L, 41)], qx = 1.0)
    m3 = model2(P = z -> 1000e3 * (1 - z / L), qx = 1.0)
    @test m1.P ≈ m2.P ≈ m3.P
    @test solve_beam_column(m1).d ≈ solve_beam_column(m3).d
    @test m1.P[1] == 1000e3 && m1.P[end] == 0.0

    # Independent sine-basis Galerkin reference for the u equation with linearly
    # varying P(z) = α + βz (the weak form ∫ P δu' u' dz couples the modes):
    #   ∫₀ᴸ z cos(λₘz) cos(λₙz) dz = L²/4 (m = n), -(L²/π²)[1/(m-n)² + 1/(m+n)²] (m±n odd), 0 else
    function spectral_u(z; P1, P2, q, N = 200)
        EI = mat.E * sec2.Iy
        α, β = P1, (P2 - P1) / L
        A = zeros(N, N); b = zeros(N)
        for mm in 1:N
            λm = mm * π / L
            for n in 1:N
                λn = n * π / L
                Z = mm == n ? L^2 / 4 :
                    isodd(mm - n) ? -(L^2 / π^2) * (1 / (mm - n)^2 + 1 / (mm + n)^2) : 0.0
                C = (mm == n ? α * L / 2 : 0.0) + β * Z
                A[mm, n] = (mm == n ? EI * λm^4 * L / 2 : 0.0) - λm * λn * C
            end
            b[mm] = q * L * (1 - (-1)^mm) / (mm * π)
        end
        x = A \ b
        return sum(x[n] * sin(n * π * z / L) for n in 1:N)
    end
    for (P1, P2) in ((1000e3, 0.0), (500e3, 1500e3), (-800e3, 1200e3))
        sol = solve_beam_column(model2(P = (P1, P2), qx = 1.0))
        for z in (0.25L, 0.5L, 0.75L)
            @test lateral_deflection(sol, z) ≈ spectral_u(z; P1, P2, q = 1.0) rtol = 1e-4
        end
    end
    # a linearly varying P shifts the peak deflection toward the more compressed end
    sol = solve_beam_column(model2(P = (1500e3, 0.0), qx = 1.0))
    zs = range(0, L, 2439)
    @test zs[argmax([lateral_deflection(sol, z) for z in zs])] < 0.5L
    # critical load factor of the triangular distribution exceeds the uniform one
    @test critical_load_factors(model2(P = (1.0, 0.0))).factors[1] > Pe_y(sec2)
end

@testset "element end forces" begin
    # continuity across element boundaries (no braces/supports inside)
    for m in (model2(P = 800e3, qx = 0.5, qy = 1.0, imperfection = imp,
                     restraint = Restraint(kx = 0.05, kϕ = 100.0, hy = 225.0)),
              model1(P = 8e3, qy = 0.02, imperfection = imp, bc_right = :fixed,
                     restraint = Restraint(kϕ = 0.1)))
        f = element_end_forces(solve_beam_column(m))
        for e in 1:(PBC.nelements(m) - 1), a in (:Vx, :My, :Vy, :Mx, :T, :B)
            x2, x1 = getfield(f[e].end2, a), getfield(f[e + 1].end1, a)
            ref = maximum(abs, [getfield(f[k].end2, a) for k in 1:PBC.nelements(m)])
            @test isapprox(x2, x1; rtol = 1e-6, atol = 1e-8 * ref + 1e-12)
        end
    end
    # pinned ends are free to warp: zero bimoment; fixed ends restrain warping
    f = element_end_forces(solve_beam_column(model2(P = 1000e3, imperfection = imp, bc_right = :fixed,
                                                    braces = [Brace(L / 2; kx = 10e3, hy = 225.0)])))
    Bmax = maximum(abs(fe.end2.B) for fe in f)
    @test abs(f[1].end1.B) < 1e-8 * Bmax
    @test abs(f[end].end2.B) > 0.1Bmax
    # the internal transverse force drops across a braced node by the brace force
    # (the brace pulls back on the member) ...
    sol = solve_beam_column(model2(P = 1000e3, imperfection = imp, braces = [Brace(L / 2; kx = 10e3, hy = 225.0)]))
    f = element_end_forces(sol)
    bf = brace_forces(sol)[1]
    @test f[21].end1.Vx - f[20].end2.Vx ≈ bf.Fx rtol = 1e-8
    # ... and the torque by the brace force times its lever arm eu = yo - hy = -225
    @test f[21].end1.T - f[20].end2.T ≈ -225.0 * bf.Fx rtol = 1e-8
    # vertical statics unaffected by P for the doubly symmetric section
    f = element_end_forces(solve_beam_column(model2(P = 1000e3, qy = 1.0)))
    @test f[1].end1.Vy ≈ L / 2 rtol = 1e-8
    @test f[20].end2.Mx ≈ -(1.0 / (1000e3 / (mat.E * sec2.Ix))) * (sec(sqrt(1000e3 / (mat.E * sec2.Ix)) * L / 2) - 1) rtol = 1e-6
    # a support with free twist: reactions at supports show up as jumps in Vy
    m = BeamColumnModel(L = 2L, nel = 40, section = sec2, material = mat, qy = 1.0,
                        supports = [Support(L)])
    f = element_end_forces(solve_beam_column(m))
    @test f[20].end2.Vy - f[21].end1.Vy ≈ -(5 / 4) * L rtol = 1e-8      # central reaction of a two-span beam: 5qL/4 (q=1, span L)
end

@testset "supports, end conditions and non-uniform meshes" begin
    # two-span column with an interior pinned support ≡ pinned-fixed single span (symmetry)
    r = Restraint(kx = 0.05, kϕ = 100.0, hy = 225.0)
    two = BeamColumnModel(L = 2L, nel = 80, section = sec2, material = mat, restraint = r,
                          P = 1000e3, qx = 0.5, supports = [Support(L)])
    sol2 = solve_beam_column(two)
    solpf = solve_beam_column(model2(P = 1000e3, qx = 0.5, restraint = r, bc_right = :fixed))
    for z in (0.2L, 0.5L, 0.8L)
        @test lateral_deflection(sol2, z) ≈ lateral_deflection(solpf, z) rtol = 1e-8
        @test twist(sol2, z) ≈ twist(solpf, z) rtol = 1e-8 atol = 1e-12
        @test lateral_deflection(sol2, z) ≈ lateral_deflection(sol2, 2L - z) rtol = 1e-8
    end
    # end support via Support(...) with :free end conditions ≡ :pinned
    ma = model2(P = 1000e3, qx = 0.5, imperfection = imp, restraint = r)
    mb = model2(P = 1000e3, qx = 0.5, imperfection = imp, restraint = r, bc_left = :free, bc_right = :free,
                supports = [Support(0.0), Support(L)])
    @test solve_beam_column(ma).d ≈ solve_beam_column(mb).d
    # per-DOF fixity: Support(z; u′ = true, v′ = true, ϕ′ = true) on a pinned end ≡ :fixed
    mc = model2(P = 1000e3, qx = 0.5, imperfection = imp, restraint = r,
                supports = [Support(L; u′ = true, v′ = true, ϕ′ = true)])
    md = model2(P = 1000e3, qx = 0.5, imperfection = imp, restraint = r, bc_right = :fixed)
    @test solve_beam_column(mc).d ≈ solve_beam_column(md).d
    # non-uniform mesh: same converged answer as the uniform mesh
    zn = sort(unique(vcat(range(0, L, 21), L / 2 .+ 60.0 .* (-4:4), 0.2L .+ 15.0 .* (-3:3))))
    mn = BeamColumnModel(z = zn, section = sec2, material = mat, P = 1000e3, imperfection = imp,
                         braces = [Brace(zn[argmin(abs.(zn .- L / 2))]; kx = 10e3, hy = 225.0)])
    mu = model2(nel = 80, P = 1000e3, imperfection = imp, braces = [Brace(L / 2; kx = 10e3, hy = 225.0)])
    @test nodes(mn) == zn
    @test twist(solve_beam_column(mn), L / 2) ≈ twist(solve_beam_column(mu), L / 2) rtol = 1e-4
    # per-element property vectors: a stepped member with the same section everywhere
    ms = BeamColumnModel(L = L, nel = 40, section = fill(sec2, 40), material = fill(mat, 40),
                         restraint = fill(r, 40), P = 1000e3, qx = 0.5)
    @test solve_beam_column(ms).d ≈ solve_beam_column(model2(P = 1000e3, qx = 0.5, restraint = r)).d
    # a stiffer half stiffens the response
    stiff = Section(A = sec2.A, Ix = sec2.Ix, Iy = 4sec2.Iy, J = sec2.J, Cw = sec2.Cw, Io = sec2.Io)
    mh = BeamColumnModel(L = L, nel = 40, section = vcat(fill(stiff, 20), fill(sec2, 20)),
                         material = mat, P = 1000e3, qx = 0.5)
    @test 0 < lateral_deflection(solve_beam_column(mh), L / 2) <
          lateral_deflection(solve_beam_column(model2(P = 1000e3, qx = 0.5)), L / 2)
end

@testset "mesh convergence" begin
    ϕ = [twist(solve_beam_column(model1(nel = n, P = 10e3, imperfection = imp,
                                        braces = [Brace(0.2L; kϕ = 500e3)])), 0.5L)
         for n in (10, 20, 40, 80)]
    err = abs.(ϕ[1:3] .- ϕ[4])
    @test err[2] < err[1] && err[3] < err[2]
    @test abs(ϕ[3] - ϕ[4]) / abs(ϕ[4]) < 1e-4
    Pcr = [critical_load_factors(model1(nel = n, P = 1.0)).factors[1] for n in (4, 8, 16, 32)]
    @test all(Pcr .>= Pe_ft(sec1) * (1 - 1e-12))          # displacement FE bounds Pcr from above
    @test issorted(Pcr; rev = true)
end

@testset "stability flag and input validation" begin
    # beyond the critical load the tangent stiffness is indefinite
    sol = @test_logs (:warn, r"not positive definite") solve_beam_column(model2(P = 1.1Pe_y(sec2), imperfection = imp))
    @test !sol.stable
    @test solve_beam_column(model2(P = 0.9Pe_y(sec2), imperfection = imp)).stable
    @test_logs solve_beam_column(model2(P = 1.1Pe_y(sec2), imperfection = imp); warn = false)

    @test_throws ArgumentError model2(bc_left = :clamped)
    @test_throws ArgumentError model2(braces = [Brace(L / 2 + 13.0; kϕ = 1.0)])
    @test_throws ArgumentError model2(supports = [Support(L / 2 + 13.0)])
    @test_throws ArgumentError BeamColumnModel(L = L, nel = 40, z = [0.0, 1.0, 2.0], section = sec2, material = mat)
    @test_throws ArgumentError BeamColumnModel(section = sec2, material = mat)
    @test_throws ArgumentError BeamColumnModel(z = [0.0, 2.0, 1.0], section = sec2, material = mat)
    @test_throws ArgumentError BeamColumnModel(L = L, nel = 40, section = fill(sec2, 39), material = mat)
    @test_throws ArgumentError model2(P = ones(7))
    @test_throws ArgumentError Section(A = -1.0, Ix = 1.0, Iy = 1.0, J = 1.0, Cw = 1.0)
    @test_throws ArgumentError critical_load_factors(model2(qy = 1.0))
    sol = solve_beam_column(model2(P = 100e3, qx = 1.0))
    @test_throws DomainError twist(sol, -1.0)
    @test sol.residual_norm < 1e-8
    @test deflections(sol, L / 2) == (u = lateral_deflection(sol, L / 2), v = vertical_deflection(sol, L / 2), ϕ = twist(sol, L / 2))
end

end
