"""
Magnetic advection-diffusion eigenvalues on an L-shaped domain: classical
magnetic-field (B) formulation vs. the vector-potential (A) formulation of
Yizhou Liang's note (notes/note2_Yizu.pdf).

Both problems are posed on the same flat L-shape, but the fields are kept as
genuine 3-component vectors that are independent of z ("2.5D").  This is the
only way to realise BOTH H(curl) formulations faithfully on a flat domain,
because on a flat 2D mesh A and B = curl A cannot both be in-plane H(curl)
vectors (the curl of an in-plane vector is a scalar).  A z-independent
3-vector field splits as

    F = (F_t, F_3),   F_t = (F_1, F_2) in-plane (N1curl),   F_3 out-of-plane (CG)

and for such a field

    curl F = ( d_y F_3,  -d_x F_3,  d_x F_2 - d_y F_1 )
           = ( gradperp(F_3),       scurl(F_t)        ).

Classical magnetic-field problem (notes, section 1).  Find B in H_0(curl),
div B = 0, and lambda such that

    curl curl B + curl(u x B) = lambda B.

Weak mixed form with multiplier psi in H^1_0 (enforces div B = 0, B.n free):

    (1/Re)(curl B, curl v) + (u x B, curl v) + (grad psi, v) = lambda (B, v)
    (B, grad phi) = 0.

Essential BCs: tangential B = 0 and B_3 = 0 on the boundary (i.e. n x B = 0),
plus psi = 0 on the boundary.

Vector-potential problem (notes, mixed formulation of A).  With B = curl A,
gauge div A = 0, find A in H(curl) and lambda such that

    (curl A, curl sigma) + (u x curl A, sigma) - (grad psi, sigma)
        + nu (A, sigma) = (lambda + nu)(A, sigma)
    (A, grad phi) = 0,   phi in H^1/C.

Here A carries no essential BC: n x curl A = 0 (i.e. n x B = 0) is natural, and
psi in H^1/C carries a constant null space.  The shift nu only regularises the
solution operator; it cancels at the eigenvalue level and we subtract it back.

Because B = curl A, the two nonzero spectra coincide.  This script computes
both and reports the pairwise agreement.

It also compares the *pseudospectra* of the two formulations.  The spectrum is
the same, but the pseudospectrum is norm-dependent, and the two formulations
carry different natural norms (L^2 on B versus L^2 on A = L^2 on curl^{-1} B).
On the leading k-mode invariant subspace the non-normality is captured by the
Gram matrix G_ij = <phi_i, phi_j> of the eigenmodes in that norm; with a diag-
onal eigenvalue matrix D the reduced resolvent norm is

    || (z - L)^{-1} ||  =  || R diag(1/(z - lambda_i)) R^{-1} ||_2,   R = G^{1/2},

whose 1/eps level sets are the eps-pseudospectrum.  G = I (orthonormal modes)
gives the normal case (disks around eigenvalues); off-diagonal mass inflates
the contours.  The graph overlays the two formulations so any difference shows.

Options (all PETSc):
    -Re        inverse diffusivity            (default 1.0)
    -maxh      netgen target mesh size        (default 0.1)
    -neig      number of eigenpairs compared  (default 12)
    -wind      discontinuous-wind selector    (default 1, see catalogue below)
    -target    shift-invert target (real)     (default 10.0)
    -nu        A-formulation shift            (default 0.0)
    -pseudo    draw the pseudospectra graph   (default True)
    -pgrid     pseudospectra grid resolution  (default 200)
    -ppad      complex-plane padding fraction (default 0.6)
    -plevels   number of sigma_min level sets (default 32)
    -plog      log-spaced levels near eigs    (default True)
    -pdecades  decades of sigma_min resolved  (default 3.0)
    -ptitle    draw the figure suptitle       (default True)
"""
from firedrake import *
import numpy as np
import scipy.sparse as sp
import sys, os
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

opt = PETSc.Options()
Re = Constant(opt.getReal("Re", 1.0))
maxh = opt.getReal("maxh", 0.1)
neig = opt.getInt("neig", 12)
wind_type = opt.getInt("wind", 1)
target = opt.getReal("target", 10.0)
nu = opt.getReal("nu", 0.0)
# Tiny psi-block penalty: removes the common nullvector psi=const of the A
# pencil so shift-invert does not deposit a ghost eigenvalue at the shift.
psireg = opt.getReal("psireg", 1e-8)
do_pseudo = opt.getBool("pseudo", True)
pgrid = opt.getInt("pgrid", 200)
ppad = opt.getReal("ppad", 0.6)
plevels = opt.getInt("plevels", 32)
plog = opt.getBool("plog", True)
pdecades = opt.getReal("pdecades", 3.0)
ptitle = opt.getBool("ptitle", True)

# Buffer of extra eigenpairs requested so that, after discarding the few
# spurious near-zero / multiplier modes, at least `neig` genuine ones remain.
buffer = opt.getInt("buffer", 30)
zero_tol = 1e-6          # |lambda| below this is treated as a kernel mode
n_req = neig + buffer

print("-------------------------|Parameters|-------------------------")
print(f"Domain: L-shape  [0,2]^2 \\ [1,2]^2,  netgen maxh = {maxh}")
print(f"Reynolds number: {float(Re):.2e}")
print(f"Wind type: {wind_type}    Eigenvalues compared: {neig}")
print(f"Shift-invert target: {target}    A-formulation shift nu: {nu}")
print("--------------------------------------------------------------")


# ── L-shaped mesh (netgen) ────────────────────────────────────────────────
from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
# Counter-clockwise L: unit square [0,2]^2 with the top-right square removed,
# giving a re-entrant corner at (1, 1).
pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
pids = [geo.AppendPoint(*p) for p in pts]
for i in range(len(pids)):
    geo.Append(["line", pids[i], pids[(i + 1) % len(pids)]])
mesh = Mesh(geo.GenerateMesh(maxh=maxh))
x, y = SpatialCoordinate(mesh)
print(f"Mesh: {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")

# ── Wind catalogue (in-plane, u_3 = 0) ────────────────────────────────────
#   0  zero (self-adjoint reference)
#   1  Heaviside shear            (sign(y - 1), 0)        -- jump across y = 1
#   2  cross shear                (sign(y - 1), sign(x - 1)) -- jumps on both lines
#   3  re-entrant pinwheel        sign-pattern rotating about the corner (1, 1)
#   4  smooth solenoidal swirl    curl of psi = sin(pi x) sin(pi y) (C^infty)
if wind_type == 0:
    u1, u2 = Constant(0.0), Constant(0.0)
elif wind_type == 1:
    u1 = conditional(y < 1, 1.0, -1.0)
    u2 = Constant(0.0)
elif wind_type == 2:
    u1 = conditional(y < 1, 1.0, -1.0)
    u2 = conditional(x < 1, 1.0, -1.0)
elif wind_type == 3:
    # Piecewise-constant swirl around the re-entrant corner (1, 1):
    # discontinuous across both x = 1 and y = 1.
    u1 = conditional(y < 1, conditional(x < 1, 1.0, 1.0), -1.0)
    u2 = conditional(x < 1, conditional(y < 1, -1.0, 1.0), -1.0)
elif wind_type == 4:
    # Smooth divergence-free swirl: u = curl(psi) with psi = sin(pi x) sin(pi y),
    # i.e. u = (d_y psi, -d_x psi).  C^infty and NOT a gradient.
    u1 = pi * sin(pi * x) * cos(pi * y)
    u2 = -pi * cos(pi * x) * sin(pi * y)
else:
    raise ValueError(f"Unknown wind type: {wind_type}")
u3vec = as_vector([u1, u2, Constant(0.0)])
self_adjoint = (wind_type == 0)


# ── 2.5D helpers: assemble z-independent 3-vectors from (in-plane, scalar) ──
def full(t, s):
    """3-vector (t_1, t_2, s) from in-plane part t and out-of-plane scalar s."""
    return as_vector([t[0], t[1], s])


def curl3(t, s):
    """curl of the z-independent 3-vector (t_1, t_2, s)."""
    return as_vector([s.dx(1), -s.dx(0), t[1].dx(0) - t[0].dx(1)])


# Shift-invert eigensolver options shared by both formulations.  MUMPS with
# null-pivot detection (icntl_24) absorbs the singular multiplier block.
opts = {
    "eps_gen_non_hermitian": None,
    "eps_target": target,
    "eps_target_magnitude": None,
    "st_type": "sinvert",
    "st_pc_type": "lu",
    "st_pc_factor_mat_solver_type": "mumps",
    "st_mat_mumps_icntl_24": 1,
    "st_mat_mumps_icntl_25": 0,
}


def mat_to_scipy(petscmat):
    """Convert an assembled PETSc Mat to a scipy CSR matrix (complex)."""
    ai, aj, av = petscmat.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=petscmat.getSize())


def collect_eigs(eigensolver, nconv, shift=0.0):
    """Genuine (nonzero, finite) eigenpairs (lambda, monolithic mode vector),
    sorted by real part.  The mode vector is the full mixed-space DOF array."""
    pairs = []
    for k in range(nconv):
        lam = complex(eigensolver.eigenvalue(k)) - shift
        if abs(lam) < zero_tol or abs(lam) > 1e8:
            continue
        er, _ = eigensolver.eigenfunction(k)   # complex build: er holds the mode
        with er.dat.vec_ro as pv:
            mode = pv.getArray().copy()
        pairs.append((lam, mode))
    pairs.sort(key=lambda p: (p[0].real, p[0].imag))
    return pairs


def solve_B():
    """Classical magnetic-field formulation.  Returns (pairs, N) where N is the
    L^2(B) Gram matrix (scipy) on the mixed space (psi block is zero)."""
    V = FunctionSpace(mesh, "N1curl", 1)
    Q = FunctionSpace(mesh, "CG", 1)
    X = MixedFunctionSpace([V, Q, Q])      # [B_t, B_3, psi]
    Bt, B3, psi = TrialFunctions(X)
    vt, v3, q = TestFunctions(X)

    B = full(Bt, B3)
    v = full(vt, v3)
    cB = curl3(Bt, B3)
    cv = curl3(vt, v3)

    a = (1 / Re) * inner(cB, cv) * dx
    if not self_adjoint:
        a += inner(cross(u3vec, B), cv) * dx
    a += inner(grad(psi), vt) * dx + inner(Bt, grad(q)) * dx
    m = inner(B, v) * dx

    # n x B = 0  ->  tangential B_t = 0 and B_3 = 0;  H^1_0 multiplier psi = 0.
    bcs = [
        DirichletBC(X.sub(0), Constant((0.0, 0.0)), "on_boundary"),
        DirichletBC(X.sub(1), Constant(0.0), "on_boundary"),
        DirichletBC(X.sub(2), Constant(0.0), "on_boundary"),
    ]
    prob = LinearEigenproblem(A=a, M=m, bcs=bcs)
    solver = LinearEigensolver(prob, n_evals=n_req, solver_parameters=opts,
                               options_prefix="")
    nconv = solver.solve()
    pairs = collect_eigs(solver, nconv)
    # L^2(B) norm matrix: ||B||^2 = (B, B).
    Nmat = assemble(inner(B, v) * dx).petscmat
    return pairs, mat_to_scipy(Nmat)


def solve_A():
    """Vector-potential formulation (Yizhou's note).  Returns (pairs, N) where N
    is the L^2(A) Gram matrix (scipy), the norm in which the note analyses
    T: L^2 -> L^2."""
    V = FunctionSpace(mesh, "N1curl", 1)
    Q = FunctionSpace(mesh, "CG", 1)
    X = MixedFunctionSpace([V, Q, Q])      # [A_t, A_3, psi]
    At, A3, psi = TrialFunctions(X)
    st, s3, q = TestFunctions(X)

    A = full(At, A3)
    sig = full(st, s3)
    cA = curl3(At, A3)
    csig = curl3(st, s3)

    a = (1 / Re) * inner(cA, csig) * dx
    if not self_adjoint:
        a += inner(cross(u3vec, cA), sig) * dx
    a += -inner(grad(psi), st) * dx + inner(At, grad(q)) * dx
    if psireg != 0.0:
        # Regularise the psi=const nullvector: the multiplier modes move to
        # lambda = infinity (filtered) instead of producing a shift-tracking
        # ghost via MUMPS null-pivot detection.
        a += psireg * inner(psi, q) * dx
    if nu != 0.0:
        a += nu * inner(A, sig) * dx      # shift; cancels with (lambda+nu) below
    m = inner(A, sig) * dx

    # A is free (n x curl A = 0 is natural).  psi in H^1/C: no boundary BC, the
    # constant mode is a genuine null vector (absorbed by MUMPS icntl_24).
    prob = LinearEigenproblem(A=a, M=m)
    solver = LinearEigensolver(prob, n_evals=n_req, solver_parameters=opts,
                               options_prefix="")
    nconv = solver.solve()
    pairs = collect_eigs(solver, nconv, shift=nu)
    # L^2(A) norm matrix: ||A||^2 = (A, A).
    Nmat = assemble(inner(A, sig) * dx).petscmat
    return pairs, mat_to_scipy(Nmat)


print("Solving classical magnetic-field (B) formulation ...")
pairs_B, Nsp_B = solve_B()
print("Solving vector-potential (A) formulation ...")
pairs_A, Nsp_A = solve_A()
eigs_B = [p[0] for p in pairs_B]
eigs_A = [p[0] for p in pairs_A]


# ── Comparison ─────────────────────────────────────────────────────────────
def match(target_list, pool):
    """Nearest-neighbour match of each value in target_list against pool."""
    out = []
    for lam in target_list:
        j = min(range(len(pool)), key=lambda i: abs(pool[i] - lam))
        out.append((lam, pool[j], abs(pool[j] - lam)))
    return out

ncmp = min(neig, len(eigs_B), len(eigs_A))
print()
print("=================== B  vs.  A  eigenvalues ====================")
print(f"{'k':>2}  {'B-formulation':>26}  {'A-formulation':>26}  {'|diff|':>9}")
max_err = 0.0
for k, (lb, la, err) in enumerate(match(eigs_B[:ncmp], eigs_A)):
    max_err = max(max_err, err)
    print(f"{k:>2}  {lb.real:>12.6e}{lb.imag:>+12.6e}i  "
          f"{la.real:>12.6e}{la.imag:>+12.6e}i  {err:>9.2e}")
print("--------------------------------------------------------------")
print(f"converged eigenvalues:  B = {len(eigs_B)},  A = {len(eigs_A)}")
print(f"max |B - A| over {ncmp} matched pairs: {max_err:.3e}")
print("==============================================================")

import pickle
run_tag = f"Re{float(Re):.3e}_wind{wind_type}_maxh{maxh:g}_neig{neig}"
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output")
os.makedirs(out_dir, exist_ok=True)
pkl_path = os.path.join(out_dir, f"mhd_BvsA_lshape_{run_tag}.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump({"Re": float(Re), "wind": wind_type, "maxh": maxh,
                 "eigs_B": eigs_B, "eigs_A": eigs_A, "max_err": max_err}, f)
print(f"Eigenvalue data written to {pkl_path}")


# ── Pseudospectra comparison ───────────────────────────────────────────────
def reduced_model(pairs, Nsp, k):
    """Diagonal eigenvalues D and unit-diagonal Gram matrix G of the leading k
    modes in the L^2 inner product encoded by the sparse matrix Nsp."""
    lams = np.array([p[0] for p in pairs[:k]])
    Phi = np.column_stack([p[1] for p in pairs[:k]])     # n x k complex
    G = Phi.conj().T @ (Nsp @ Phi)                       # k x k Hermitian PD
    G = 0.5 * (G + G.conj().T)                           # clean rounding
    d = np.sqrt(np.real(np.diag(G)))
    G = G / np.outer(d, d)                               # unit diagonal (angles)
    return lams, G


def sigma_min_field(lams, G, xs, ys):
    """Smallest singular value sigma_min(z - L) of the reduced resolvent over
    the grid xs x ys, in the G-inner-product norm with weight R = G^{1/2}.
    This is the reciprocal of the reduced resolvent norm, so its 1/eps level
    set eps is the boundary of the eps-pseudospectrum; small values mark the
    spectrum.  Mirrors the sigma_min(A - z M) field used in the paper plates."""
    w, U = np.linalg.eigh(G)
    w = np.clip(w, 1e-14, None)
    R = (U * np.sqrt(w)) @ U.conj().T
    Rinv = (U * (1.0 / np.sqrt(w))) @ U.conj().T
    field = np.empty((ys.size, xs.size))
    for iy, zy in enumerate(ys):
        for ix, zx in enumerate(xs):
            diag = 1.0 / ((zx + 1j * zy) - lams)
            M = (R * diag) @ Rinv                        # R diag(diag) R^{-1}
            field[iy, ix] = 1.0 / np.linalg.norm(M, 2)   # sigma_min(z - L)
    return field


if do_pseudo and PETSc.COMM_WORLD.getRank() == 0:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    # "Mana (Extended)" colour map, identical to the paper's render scripts so
    # this plate matches figures/render_pseudospectra_2D.py.
    _PV_MANA_STOPS = [
        (0.00000, (0.098039, 0.137255, 0.352941)),
        (0.03125, (0.207283, 0.138936, 0.373669)),
        (0.06250, (0.316527, 0.140616, 0.394398)),
        (0.09375, (0.425770, 0.142297, 0.415126)),
        (0.12500, (0.535014, 0.143978, 0.435854)),
        (0.15625, (0.644258, 0.145658, 0.456583)),
        (0.18750, (0.753501, 0.147339, 0.477311)),
        (0.21875, (0.862745, 0.149020, 0.498039)),
        (0.25000, (0.803922, 0.200490, 0.560784)),
        (0.28125, (0.745098, 0.251961, 0.623529)),
        (0.31250, (0.686275, 0.303431, 0.686275)),
        (0.34375, (0.627451, 0.354902, 0.749020)),
        (0.37500, (0.568627, 0.406373, 0.811765)),
        (0.40625, (0.509804, 0.457843, 0.874510)),
        (0.43750, (0.450980, 0.509314, 0.937255)),
        (0.46875, (0.392157, 0.560784, 1.000000)),
        (0.50000, (0.343137, 0.611765, 0.916667)),
        (0.53125, (0.294118, 0.662745, 0.833333)),
        (0.56250, (0.245098, 0.713725, 0.750000)),
        (0.59375, (0.196078, 0.764706, 0.666667)),
        (0.62500, (0.464052, 0.797386, 0.699346)),
        (0.65625, (0.732026, 0.830065, 0.732026)),
        (0.68750, (1.000000, 0.862745, 0.764706)),
        (0.71875, (0.993464, 0.790850, 0.679739)),
        (0.75000, (0.986928, 0.718954, 0.594771)),
        (0.78125, (0.980392, 0.647059, 0.509804)),
        (0.81250, (0.933333, 0.560784, 0.435294)),
        (0.84375, (0.886275, 0.474510, 0.360784)),
        (0.87500, (0.839216, 0.388235, 0.286275)),
        (0.90625, (0.729412, 0.291176, 0.245098)),
        (0.93750, (0.619608, 0.194118, 0.203922)),
        (0.96875, (0.509804, 0.097059, 0.162745)),
        (1.00000, (0.400000, 0.000000, 0.121569)),
    ]
    PV_MANA = LinearSegmentedColormap.from_list("mana", _PV_MANA_STOPS, N=256)
    DPI = 300

    kp = min(neig, len(pairs_B), len(pairs_A))
    lams_B, G_B = reduced_model(pairs_B, Nsp_B, kp)
    lams_A, G_A = reduced_model(pairs_A, Nsp_A, kp)

    # Off-diagonal mass = a scalar summary of non-normality in each norm.
    offdiag = lambda G: np.linalg.norm(G - np.diag(np.diag(G)))
    print(f"Non-normality (||G - I||_F):  B = {offdiag(G_B):.3e}   "
          f"A = {offdiag(G_A):.3e}")

    # Shared complex-plane window around the eigenvalues.
    allre = np.concatenate([lams_B.real, lams_A.real])
    allim = np.concatenate([lams_B.imag, lams_A.imag])
    rspan = max(float(np.ptp(allre)), 1.0)
    ispan = max(float(np.ptp(allim)), 1.0)
    xs = np.linspace(allre.min() - ppad * rspan, allre.max() + ppad * rspan, pgrid)
    ys = np.linspace(allim.min() - ppad * ispan, allim.max() + ppad * ispan, pgrid)

    print(f"Sampling pseudospectra on a {pgrid}x{pgrid} grid ...")
    F_B = sigma_min_field(lams_B, G_B, xs, ys)
    F_A = sigma_min_field(lams_A, G_A, xs, ys)

    # Shared colour range over both panels.
    vmax = float(max(F_B.max(), F_A.max()))
    fmin = float(min(F_B.min(), F_A.min()))
    if plog:
        # sigma_min -> 0 at the eigenvalues, so log-spaced level sets resolve
        # the tight closed loops near each eigenvalue.  The floor is capped at
        # vmax * 10^-pdecades so a grid point that lands almost on top of an
        # eigenvalue does not swallow the whole dynamic range.
        vmin = max(fmin, vmax * 10.0 ** (-pdecades))
        if not (vmin > 0.0 and vmax > vmin):
            vmin, vmax = max(vmax * 1e-3, 1e-30), max(vmax, 1e-29)
        levels = np.geomspace(vmin, vmax, plevels + 1)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        iso_levels = levels[1:-1]            # every level set drawn as a curve
    else:
        vmin = fmin
        if vmax - vmin < 1e-14 * max(1.0, abs(vmax)):
            vmax = vmin + 1.0
        levels = np.linspace(vmin, vmax, plevels + 1)
        norm = None
        iso_levels = np.linspace(vmin, vmax, 9)[1:-1]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), constrained_layout=True)
    for ax, F, lams, ttl in (
        (axes[0], F_B, lams_B, r"B-formulation  ($L^2(B)$ norm)"),
        (axes[1], F_A, lams_A, r"A-formulation  ($L^2(A)$ norm)"),
    ):
        Fc = np.clip(F, vmin, vmax)
        cf = ax.contourf(xs, ys, Fc, levels=levels, norm=norm,
                         vmin=vmin, vmax=vmax, cmap=PV_MANA, extend="neither")
        ax.contour(xs, ys, Fc, levels=iso_levels, colors="k",
                   linewidths=0.3, alpha=0.45)
        # Compatible discretisation (N1curl x CG) -> white dots, black edge.
        ax.scatter(lams.real, lams.imag, s=28, marker="o",
                   facecolors="white", edgecolors="black",
                   linewidths=1.2, zorder=5)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel(r"$\operatorname{Re}(z)$")
        ax.set_ylabel(r"$\operatorname{Im}(z)$")
        ax.tick_params(direction="in", which="both")
    cbar = fig.colorbar(cf, ax=axes, pad=0.02, extend="neither")
    cbar.set_label(r"$\sigma_{\min}(z - L)$")
    if plog:
        # Explicit decade ticks (10^-3, 10^-2, ...): the discrete contour
        # colorbar ignores a LogLocator, so we set the tick values directly.
        lo = int(np.ceil(np.log10(vmin)))
        hi = int(np.floor(np.log10(vmax)))
        exps = list(range(lo, hi + 1))
        if exps:
            cbar.set_ticks([10.0 ** e for e in exps])
            cbar.set_ticklabels([rf"$10^{{{e}}}$" for e in exps])

    if ptitle:
        fig.suptitle(
            f"Pseudospectra: B vs A formulation  "
            f"(L-shape, $\\mathrm{{Rm}}={float(Re):.1e}$, wind {wind_type}, "
            f"maxh={maxh:g}, $k={kp}$)",
            fontsize=11)
    out_stem = os.path.join(out_dir, f"pseudo_BvsA_lshape_{run_tag}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=DPI)
    plt.close(fig)
    print(f"Pseudospectra graph written to {out_stem}.png/.pdf")
