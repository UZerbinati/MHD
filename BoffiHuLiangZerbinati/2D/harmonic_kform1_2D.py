"""
python -Wignore -i kikuchi_advection_2D_torus.py -eps_view -eps_monitor -Re 1 -st_pc_svd_monitor -st_pc_type svd
"""
from firedrake import *
import numpy as np
import sys, os
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

opt = PETSc.Options()
Re = opt.getReal("Re", 1.0)     # Reynolds number
Re = Constant(Re)
invert_stencil = opt.getBool("invert_stencil", False)     # Invert stencil
neig = opt.getInt("neig", 100)     # Number of eigenvalue to compute
N = opt.getInt("N", 32)            # Periodic mesh resolution
geo = opt.getString("geo", "torus")     # Geometry: "torus" or "cylinder"
iterative = opt.getBool("iterative", False)
wind_type = opt.getInt("wind", 1)
# wind catalogue (canonical 6 box-mesh winds):
#   0  zero (self-adjoint)
#   1  (1, 1, 1) constant
#   2  (cos y + cos z, cos z + sin x, cos x + sin y)
#   3  (sin z + cos y, sin x + cos z, sin y + cos x)
#   4  Beltrami-like (sin x (cos y - cos z), ...)
#   5  Heaviside shear  (sign(z - pi/2), 0, 0)
if geo == "torus":
    pdir = "both"
elif geo == "cylinder":
    pdir = "x"
else:
    raise ValueError(f"Unknown geometry: {geo}. Choose 'torus' or 'cylinder'.")

print("-------------------------|Parameters|-------------------------")
print(f"Reynolds number: {float(Re):.2e}")
print(f"Invert stencil: {invert_stencil}")
print(f"Wind type: {wind_type}")
print(f"Number of eigenvalues to compute: {neig}")
print(f"Geometry: {geo}")
print(f"Mesh: {N}x{N}")
print("--------------------------------------------------------------")


mesh = PeriodicRectangleMesh(N, N, np.pi, np.pi, direction=pdir)
x, y = SpatialCoordinate(mesh)

coord_fs = VectorFunctionSpace(mesh, "CG", 1, dim=3)

if geo == "torus":
    # theta = 2*x, phi = 2*y maps [0,π]² → [0,2π]², so both directions close correctly.
    theta = 2*x
    phi = 2*y
    R = Constant(2.0)
    r = Constant(0.6)
    coords = Function(coord_fs).interpolate(as_vector((
        (R + r*cos(theta))*cos(phi),
        (R + r*cos(theta))*sin(phi),
        r*sin(theta),
    )))
    embedded = Mesh(coords)
    X_e = SpatialCoordinate(embedded)
    # Outward normal: radial direction from the tube centreline
    n = as_vector((
        X_e[0] - R*X_e[0]/sqrt(X_e[0]**2 + X_e[1]**2),
        X_e[1] - R*X_e[1]/sqrt(X_e[0]**2 + X_e[1]**2),
        X_e[2],
    ))

elif geo == "cylinder":
    r_cyl = Constant(1.0)
    # theta = 2*x maps x∈[0,π] → theta∈[0,2π], so the cylinder closes correctly.
    # (The shared periodic node at x=0/x=π gives cos(0)=cos(2π)=1.)
    theta_cyl = 2*x
    coords = Function(coord_fs).interpolate(as_vector((
        r_cyl*cos(theta_cyl),
        r_cyl*sin(theta_cyl),
        y,                      # axial coordinate, runs over [0, π]
    )))
    embedded = Mesh(coords)
    X_e = SpatialCoordinate(embedded)
    # Outward normal: radially away from the cylinder axis
    n = as_vector((X_e[0], X_e[1], Constant(0.0)))
else:
    raise ValueError(f"Unknown geometry: {geo}. Choose 'torus' or 'cylinder'.")

embedded.init_cell_orientations(n)
mesh = embedded

x, y,z = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "N1curl", 1)
Q = FunctionSpace(mesh, "CG", 1)
X = MixedFunctionSpace([V, Q])
u,psi = TrialFunctions(X)
v,phi = TestFunctions(X)

if wind_type == 0:
    u_wind = as_vector([0.0, 0.0, 0.0])
elif wind_type == 1:
    u_wind = as_vector([1.0, 1.0, 1.0])
elif wind_type == 2:
    u_wind = as_vector([cos(y) + cos(z), cos(z) + sin(x), cos(x) + sin(y)])
elif wind_type == 3:
    u_wind = as_vector([sin(z) + cos(y), sin(x) + cos(z), sin(y) + cos(x)])
elif wind_type == 4:
    # curl A with A = (sin y sin z, sin z sin x, sin x sin y); ambient
    # divergence-free Beltrami-like flow, NOT a gradient.
    u_wind = as_vector([
        sin(x)*(cos(y) - cos(z)),
        sin(y)*(cos(z) - cos(x)),
        sin(z)*(cos(x) - cos(y)),
    ])
elif wind_type == 5:
    # Discontinuous Heaviside shear in the ambient z direction.
    u_wind = as_vector([conditional(z < pi/2, 1.0, -1.0),
                        Constant(0.0), Constant(0.0)])
else:
    raise ValueError(f"Unknown wind type: {wind_type}")
self_adjoint = (wind_type == 0)

#Adding the null space assocciated with the Lagrange multiplier
v_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(X, [v_basis, X.sub(1)])
nullspace._build_monolithic_basis()

boundary_conditions = [
        DirichletBC(X.sub(0), Constant((0.0, 0.0, 0.0)), "on_boundary"),
        DirichletBC(X.sub(1), Constant(0.0), "on_boundary")
]

a = (1/Re)*inner(curl(u), curl(v))*dx
if not self_adjoint:
    a += inner(cross(u_wind, u), curl(v))*dx
a += inner(grad(psi), v)*dx + inner(u, grad(phi))*dx
m = inner(u, v)*dx

A = assemble(a)
M = assemble(m)

A.petscmat.setNullSpace(nullspace._nullspace)
M.petscmat.setNullSpace(nullspace._nullspace)

A.petscmat.assemble()
M.petscmat.assemble()

# Riesz inner product on V x Q = N1curl x CG (H(curl) x H^1).  The L^2
# mass term on psi makes B SPD on the whole space (the constant-psi
# nullspace of A, M is *not* in B's kernel), which is what gamg expects
# on the multiplier block.
b = (inner(u, v)*dx + inner(curl(u), curl(v))*dx
     + inner(psi, phi)*dx + inner(grad(psi), grad(phi))*dx)

opts = {"eps_gen_non_hermitian": None,
        "eps_target": 1e-8,        # shift-invert around 0 to find near-zero eigs
        "eps_target_real": None,  # rank by real part among candidates near target
        "st_type": "sinvert",
        # The constant-psi nullspace makes (A - sigma M) singular; MUMPS LU
        # with ICNTL(24)=1 detects the null pivot rows and returns a valid
        # solution.  PCQR (SuiteSparseQR) handles this too but is serial-only,
        # so it OOMs / fails under mpiexec.
        "st_pc_type": "lu",
        "st_pc_factor_mat_solver_type": "mumps",
        "st_mat_mumps_icntl_24": 1,
        "st_mat_mumps_icntl_25": 0,
}

data = {"Re": float(Re),
        "eigenvalues": [],
        "normalized_eigenvalues": []
}


print("Solving ...")
if iterative:
    if invert_stencil:
        raise NotImplementedError(
            "iterative path does not support -invert_stencil; the Riesz "
            "preconditioner targets (A - sigma M), not the swapped pencil."
        )
    print("(iterative GMRES + Riesz H(curl)xH^1 preconditioner: "
          "fieldsplit-additive vertex-star ASM on N1curl + gamg on CG)")
    from riesz_solver import IterativeEigensolver
    from riesz_pcs import HARMONIC_KFORM1_2D as RIESZ_PC
    B_mat = assemble(b, bcs=boundary_conditions, mat_type="aij")
    M_it = assemble(m, bcs=boundary_conditions, weight=0.0)
    eigensolver = IterativeEigensolver(
        A.petscmat, M_it.petscmat, B_mat.petscmat,
        function_space=X, n_evals=neig,
        target=1e-8, which="TARGET_MAGNITUDE", problem_type="GNHEP",
        pc_solver_parameters=RIESZ_PC,
    )
else:
    if invert_stencil:
        eigenproblem = LinearEigenproblem(A=m, M=a)
    else:
        eigenproblem = LinearEigenproblem(A=a, M=m)
    eigensolver = LinearEigensolver(eigenproblem,
                                    n_evals=neig,
                                    solver_parameters=opts,
                                    options_prefix="")
nconv = eigensolver.solve()

run_tag = (
    f"{geo}_Re{float(Re):.3e}"
    f"_wind{wind_type}_invert-{int(invert_stencil)}"
    f"_neig{neig}"
)
eigs_path = f"../output/eigs_{run_tag}.pvd"
fp = File(eigs_path, mode="w")
u = Function(V, name="B")
v = Function(Q, name="psi")
for k in range(nconv):
        if invert_stencil:
            lam = 1/eigensolver.eigenvalue(k)
        else:
            lam = eigensolver.eigenvalue(k)
        eigenmode_real, eigenmode_imag = eigensolver.eigenfunction(k)
        B, psi_f = eigenmode_real.subfunctions
        u.interpolate(B)
        v.interpolate(psi_f)

        if abs(lam) > 1e-8:
            data["eigenvalues"].append(lam)
            data["normalized_eigenvalues"].append(lam*float(Re))
            print(f"{k}-th computed eigenvalue {lam:.2e}")
        else:
            # Diagnose the type of zero eigenvalue
            norm_u    = sqrt(abs(assemble(inner(u, u)*dx)))
            norm_curl = sqrt(abs(assemble(inner(curl(u), curl(u))*dx)))
            norm_psi  = sqrt(abs(assemble(inner(v, v)*dx)))
            msg = (f"{k}-th computed eigenvalue {lam:.2e}  "
                   f"||u||={norm_u:.2e}  ||curl u||={norm_curl:.2e}  ||psi||={norm_psi:.2e}")
            if norm_u < 1e-10:
                print(GREEN % (msg + "  [pressure null space  (u≈0, psi≠0)]"))
            elif norm_curl < 1e-6 * norm_u:
                print(RED % (msg + "  [harmonic 1-form      (curl u≈0, u≠0)]"))
            else:
                print(BLUE % (msg + "  [other near-zero]"))

        fp.write(u, v, time=k)

# ── Canonical harmonic forms (torus only) ────────────────────────────────────
if geo == "torus":
    X_c = SpatialCoordinate(mesh)
    rho = sqrt(X_c[0]**2 + X_c[1]**2)
    e_phi_raw = as_vector([-X_c[1]/rho, X_c[0]/rho, Constant(0.0)])
    n_raw     = as_vector([
        X_c[0] - R*X_c[0]/rho,
        X_c[1] - R*X_c[1]/rho,
        X_c[2],
    ])
    n_unit      = n_raw / sqrt(inner(n_raw, n_raw))
    # Right-handed surface frame: e_theta = n x e_phi (poloidal unit covector).
    e_theta_raw = cross(n_unit, e_phi_raw)

    def l2(a, b):
        # In complex Firedrake assemble of a scalar form returns a complex
        # scalar; we discard the imag part deliberately because we only ever
        # call l2 with real candidates / real targets.
        return float(np.real(complex(assemble(inner(a, b)*dx))))

    # Real-basis extraction.  SLEPc in complex Firedrake returns each
    # harmonic mode multiplied by an arbitrary phase exp(i alpha), so a
    # single complex eigenfunction u_k = a_k + i b_k contributes *two* real
    # vectors {a_k, b_k} to the real harmonic subspace V_R.  Both pieces are
    # needed: dropping the imag part (the old bug) reduced span_R(...) to a
    # strict subset of V_R, which made the L^2 projections of e_theta and
    # e_phi collapse onto a single 1-d slice and come out almost identical.
    candidates = []
    for k in range(nconv):
        lam = eigensolver.eigenvalue(k)
        if abs(lam) > 1e-8:
            continue
        er, _ = eigensolver.eigenfunction(k)
        B_full, _ = er.subfunctions
        for part_name, part_data in (("Re", np.real(B_full.dat.data_ro)),
                                     ("Im", np.imag(B_full.dat.data_ro))):
            u_h = Function(V)
            u_h.dat.data[:] = part_data
            n_u = sqrt(abs(l2(u_h, u_h)))
            if n_u < 1e-10:
                # Pure pressure null-space part (u = 0): skip
                continue
            candidates.append(u_h)
    print(f"Harmonic real-basis candidates: {len(candidates)}")
    if len(candidates) < 2:
        raise RuntimeError(
            f"Found only {len(candidates)} real-valued harmonic candidates "
            "(need at least 2 to span the b_1=2 harmonic subspace of the torus)."
        )

    nc = len(candidates)
    Gm = np.zeros((nc, nc), dtype=float)
    for i in range(nc):
        for j in range(i, nc):
            Gm[i, j] = l2(candidates[i], candidates[j])
            Gm[j, i] = Gm[i, j]
    eigGm = np.linalg.eigvalsh(Gm)
    print(f"Gram spectrum (real candidates): {eigGm}")

    rhs_theta = np.array([l2(s, e_theta_raw) for s in candidates], dtype=float)
    rhs_phi   = np.array([l2(s, e_phi_raw)   for s in candidates], dtype=float)
    # Pseudoinverse: Gm has rank b_1 = 2; the extra candidates lie in its
    # kernel and contribute zero coefficients.
    Gm_pinv = np.linalg.pinv(Gm, rcond=1e-10)
    c_theta = Gm_pinv @ rhs_theta
    c_phi   = Gm_pinv @ rhs_phi
    print(f"Poloidal coefficients: {c_theta}")
    print(f"Toroidal coefficients: {c_phi}")

    harm_theta = Function(V, name="omega_theta")
    harm_phi   = Function(V, name="omega_phi")
    for i in range(nc):
        harm_theta.dat.data[:] += c_theta[i] * candidates[i].dat.data_ro
        harm_phi.dat.data[:]   += c_phi[i]   * candidates[i].dat.data_ro

    for name, h, ref in [("omega_theta", harm_theta, e_theta_raw),
                         ("omega_phi",   harm_phi,   e_phi_raw)]:
        curl_norm = sqrt(abs(l2(curl(h), curl(h))))
        l2_norm   = sqrt(abs(l2(h, h)))
        cross_pol = l2(h, e_theta_raw)
        cross_tor = l2(h, e_phi_raw)
        print(f"{name}: ||u||={l2_norm:.2e}  ||curl u||/||u||={curl_norm/max(l2_norm,1e-300):.2e}")
        print(f"  <{name}, e_theta>={cross_pol:+.3e}   <{name}, e_phi>={cross_tor:+.3e}")

    # ── Extra near-kernel mode (non-topological) ──────────────────────────
    # On a torus the topological harmonic subspace is 2-dimensional, but the
    # advection-modified operator can carry additional eigenvalues that drop
    # below the lambda~0 threshold (notably at Re=100 with the Beltrami-like
    # winds 3 and 4).  We isolate that "extra" near-kernel mode as the L^2
    # orthogonal complement of span{omega_theta, omega_phi} inside the
    # candidate span.  Saving it together with its curl magnitude makes the
    # non-harmonic character visible: a topological harmonic gives
    # |curl omega| = 0; an extra mode lights up where the wind drives flow.
    GS_BASIS = [harm_theta, harm_phi]
    extra = None
    extra_norm = 0.0
    for cand in candidates:
        r = Function(V)
        r.dat.data[:] = cand.dat.data_ro
        for b in GS_BASIS:
            bb = l2(b, b)
            if bb > 1e-300:
                r.dat.data[:] -= (l2(r, b) / bb) * b.dat.data_ro
        nr = sqrt(abs(l2(r, r)))
        if nr > extra_norm:
            extra = r
            extra_norm = nr

    omega_extra = Function(V, name="omega_extra")
    has_extra = extra is not None and extra_norm > 1e-6
    if has_extra:
        omega_extra.dat.data[:] = extra.dat.data_ro / extra_norm
        ce_pol = l2(omega_extra, e_theta_raw)
        ce_tor = l2(omega_extra, e_phi_raw)
        cnorm  = sqrt(abs(l2(curl(omega_extra), curl(omega_extra))))
        print(f"omega_extra: |residual|/|cand|={extra_norm:.2e}  "
              f"||curl||={cnorm:.2e}")
        print(f"  <omega_extra, e_theta>={ce_pol:+.3e}   "
              f"<omega_extra, e_phi>={ce_tor:+.3e}")
    else:
        print(f"omega_extra: residual = {extra_norm:.2e}  (no extra "
              "near-kernel mode beyond omega_theta, omega_phi)")

    # Curl magnitude of omega_extra as a DG0 scalar for rendering.
    DG0 = FunctionSpace(mesh, "DG", 0)
    curl_extra_mag = Function(DG0, name="curl_omega_extra_mag")
    curl_extra_mag.interpolate(sqrt(inner(curl(omega_extra), curl(omega_extra))))

    harm_path = f"../output/harmonic_forms_{run_tag}.pvd"
    fp_harm = File(harm_path, mode="w")
    fp_harm.write(harm_theta, harm_phi, omega_extra, curl_extra_mag)
    print(f"Canonical harmonic forms written to {harm_path}")

import pickle
pkl_path = f"../output/mhd_adv_potential_{run_tag}.pkl"
with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
print(f"Eigenvalue data written to {pkl_path}")
