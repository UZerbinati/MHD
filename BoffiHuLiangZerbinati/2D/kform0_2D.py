"""
2D advection-diffusion eigenvalue problem for 0-forms (full Hodge formulation).

In the dual de Rham complex on R^2:
    Lambda^0 = DG (scalar v),    Lambda^1 = RT (flux p).

The full Hodge formulation (eq:Hodge:mixed of the paper) for k=0 reads
    eps (delta^1 p, w)  +  (i_u^1 p, w)            = lambda (v, w)   forall w in DG
    (p, q)              -  (v, delta^1 q)          = 0               forall q in RT
which is the dual-mixed form of the scalar advection-diffusion eigenproblem
    -eps Delta v + u . grad v = lambda v,    v = 0 on dOmega.

Solver options (all PETSc):
    -Re        diffusivity-1 (default 1.0)
    -N         mesh size (default 32)
    -order     RT/DG polynomial order (default 1)
    -neig      number of eigenpairs (default 10)
    -wind      0 = none, 1 = (1,1), 2 = curl-flow, 3 = (sin y, sin x) (default 0)
    -pseudo_export 0/1 ; if 1 also writes a VTK pseudo-spectrum
    -npts      pseudo-spectra random samples (default 2000)
"""
from firedrake import *
import numpy as np
import sys, os
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

opt = PETSc.Options()
Re = Constant(opt.getReal("Re", 1.0))
N = opt.getInt("N", 32)
order = opt.getInt("order", 1)
neig = opt.getInt("neig", 10)
wind_type = opt.getInt("wind", 0)
pseudo_export = opt.getBool("pseudo_export", True)
npts = opt.getInt("npts", 2000)
iterative = opt.getBool("iterative", False)

print("-------------------------|Parameters|-------------------------")
print(f"k = 0 (DG x RT, full Hodge)")
print(f"Reynolds number: {float(Re):.2e}")
print(f"Mesh: {N}x{N} on [0, pi]^2,   order: {order}")
print(f"Wind type: {wind_type}    Number of eigenvalues: {neig}")
print(f"Pseudo-spectra VTK export: {pseudo_export} (npts={npts})")
print("--------------------------------------------------------------")

# Reference Dirichlet-Laplace eigenvalues on [0, pi]^2:  l^2 + m^2 (l, m >= 1)
lams_ref = sorted([l*l + m*m for l in range(1, 6) for m in range(1, 6)])[:neig]

mesh = RectangleMesh(N, N, np.pi, np.pi)
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "DG", order - 1)
P = FunctionSpace(mesh, "RT", order)
X = MixedFunctionSpace([V, P])
v, p = TrialFunctions(X)
w, q = TestFunctions(X)

if wind_type == 0:
    u_wind = as_vector([0.0, 0.0])
elif wind_type == 1:
    u_wind = as_vector([1.0, 1.0])
elif wind_type == 2:
    u_wind = as_vector([2*cos(2*x)*sin(2*y), 2*sin(2*x)*cos(2*y)])
elif wind_type == 3:
    u_wind = as_vector([sin(y), sin(x)])
elif wind_type == 4:
    # curl of stream function psi = sin(2x) sin(2y): divergence-free and
    # NOT a gradient (the scalar curl is 8 sin(2x) sin(2y), nonzero).
    u_wind = as_vector([2*sin(2*x)*cos(2*y), -2*cos(2*x)*sin(2*y)])
elif wind_type == 5:
    # Discontinuous Heaviside shear: u = (sign(y - pi/2), 0).
    u_wind = as_vector([conditional(y < pi/2, 1.0, -1.0), Constant(0.0)])
else:
    raise ValueError(f"Unknown wind type: {wind_type}")
self_adjoint = (wind_type == 0)

# Mixed Poisson + advection.  v = 0 on dOmega is the natural BC for v in DG.
a = (1/Re)*inner(div(p), w)*dx
if not self_adjoint:
    a += inner(dot(u_wind, p), w)*dx
a += inner(p, q)*dx - inner(v, div(q))*dx
m = inner(v, w)*dx

A = assemble(a, mat_type="aij")
M = assemble(m, mat_type="aij")

# Riesz inner product on V x P = DG x RT (L^2 x H(div)).
b = (inner(v, w)*dx
     + inner(p, q)*dx + inner(div(p), div(q))*dx)
B = assemble(b, mat_type="aij")

opts = {"eps_gen_non_hermitian": None,
        "eps_target": 1e-8,
        "eps_target_real": None,
        "st_type": "sinvert",
        "st_pc_type": "lu",
        "st_pc_factor_mat_solver_type": "mumps"}
if iterative:
    print("Solving eigenproblem (iterative GMRES + Riesz L^2xH(div) preconditioner: "
          "fieldsplit-additive jacobi on DG + vertex-star ASM on RT)...")
    from riesz_solver import IterativeEigensolver
    from riesz_pcs import KFORM0_2D as RIESZ_PC
    eigensolver = IterativeEigensolver(
        A.petscmat, M.petscmat, B.petscmat,
        function_space=X, n_evals=neig,
        target=1e-8, which="TARGET_MAGNITUDE", problem_type="GNHEP",
        pc_solver_parameters=RIESZ_PC,
    )
else:
    eigenproblem = LinearEigenproblem(A=a, M=m)
    print("Solving eigenproblem (direct LU/MUMPS)...")
    eigensolver = LinearEigensolver(eigenproblem, n_evals=neig,
                                    solver_parameters=opts)
nconv = eigensolver.solve()

run_tag = (
    f"N{N}_order{order}_Re{float(Re):.3e}"
    f"_wind{wind_type}_neig{neig}"
)
fp = File(f"../output/kform0_2D_{run_tag}.pvd", mode="w")
v_out = Function(V, name="v")
lams = []
for k in range(min(neig, nconv)):
    lam = eigensolver.eigenvalue(k)
    lams.append(lam)
    if abs(lam) < 1e-6:
        kind = "(near-zero, divergence-free flux mode)"
    else:
        kind = ""
    if k < len(lams_ref):
        err = abs(complex(lam) - lams_ref[k])
        print(f"{k}-th eigenvalue ref={lams_ref[k]:5.1f} -- computed {lam:.3e} (err {err:.1e}) {kind}")
    else:
        print(f"{k}-th computed eigenvalue {lam:.3e} {kind}")
    er, _ = eigensolver.eigenfunction(k)
    v_h, _ = er.subfunctions
    v_out.interpolate(v_h)
    fp.write(v_out, time=k)
print(f"Eigenmodes written to ../output/kform0_2D_{run_tag}.pvd")

if pseudo_export:
    import pseudo_tool
    prefix = f"../output/pseudo_kform0_2D_{run_tag}_npts{npts}"
    pseudo_tool.export_pseudo_spectra_vtk(
        A, M, (-2.0, 30.0, -10.0, 10.0),
        output_prefix=prefix, npts=npts, lams=lams, flip=False,
        B=B if iterative else None,
        pc_solver_parameters=(RIESZ_PC if iterative else None),
        function_space=(X if iterative else None),
    )
