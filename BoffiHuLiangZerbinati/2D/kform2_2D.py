"""
2D advection-diffusion eigenvalue problem for 2-forms (top-form, primal Galerkin).

In the dual de Rham complex on R^2:
    Lambda^2 = CG (scalar v).
There is no Lambda^3 in 2D, so no Lagrange multiplier is needed; the half-Hodge
formulation collapses to the primal Galerkin advection-diffusion eigenproblem
    -eps Delta v + u . grad v = lambda v,    v = 0 on dOmega.

Solver options as in kform0_2D.py.
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
print(f"k = 2 (CG, primal Galerkin)")
print(f"Reynolds number: {float(Re):.2e}")
print(f"Mesh: {N}x{N} on [0, pi]^2,   order: {order}")
print(f"Wind type: {wind_type}    Number of eigenvalues: {neig}")
print(f"Pseudo-spectra VTK export: {pseudo_export} (npts={npts})")
print("--------------------------------------------------------------")

lams_ref = sorted([l*l + m*m for l in range(1, 6) for m in range(1, 6)])[:neig]

mesh = RectangleMesh(N, N, np.pi, np.pi)
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", order)
v = TrialFunction(V)
w = TestFunction(V)

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

bc = DirichletBC(V, Constant(0.0), "on_boundary")

a = (1/Re)*inner(grad(v), grad(w))*dx
if not self_adjoint:
    a -= inner(v*u_wind, grad(w))*dx
m = inner(v, w)*dx

A = assemble(a, bcs=bc, mat_type="aij")
M = assemble(m, bcs=bc, mat_type="aij")

# Riesz inner product on V = CG (H^1 norm) — preconditioning matrix
# when -iterative is set.
B_form = inner(v, w)*dx + inner(grad(v), grad(w))*dx
B = assemble(B_form, bcs=bc, mat_type="aij")

if self_adjoint:
    opts = {"eps_gen_hermitian": None,
            "eps_smallest_real": None,
            "st_type": "sinvert",
            "st_pc_factor_mat_solver_type": "mumps"}
else:
    opts = {"eps_gen_non_hermitian": None,
            "eps_target": 1e-8,
            "eps_target_real": None,
            "st_type": "sinvert",
            "st_pc_type": "lu",
            "st_pc_factor_mat_solver_type": "mumps"}
if iterative:
    print("Solving eigenproblem (iterative GMRES + Riesz H^1 preconditioner: gamg)...")
    from riesz_solver import IterativeEigensolver
    from riesz_pcs import KFORM2_2D as RIESZ_PC
    problem_type = "GHEP" if self_adjoint else "GNHEP"
    M_it = assemble(m, bcs=bc, weight=0.0, mat_type="aij")
    eigensolver = IterativeEigensolver(
        A.petscmat, M_it.petscmat, B.petscmat,
        function_space=V, n_evals=neig,
        target=1e-8, which="TARGET_MAGNITUDE", problem_type=problem_type,
        pc_solver_parameters=RIESZ_PC,
    )
else:
    eigenproblem = LinearEigenproblem(A=a, M=m, bcs=bc)
    print("Solving eigenproblem (direct LU/MUMPS)...")
    eigensolver = LinearEigensolver(eigenproblem, n_evals=neig,
                                    solver_parameters=opts)
nconv = eigensolver.solve()

run_tag = (
    f"N{N}_order{order}_Re{float(Re):.3e}"
    f"_wind{wind_type}_neig{neig}"
)
fp = File(f"../output/kform2_2D_{run_tag}.pvd", mode="w")
v_out = Function(V, name="v")
lams = []
for k in range(min(neig, nconv)):
    lam = eigensolver.eigenvalue(k)
    lams.append(lam)
    if k < len(lams_ref):
        err = abs(complex(lam) - lams_ref[k])
        print(f"{k}-th eigenvalue ref={lams_ref[k]:5.1f} -- computed {lam:.3e} (err {err:.1e})")
    else:
        print(f"{k}-th computed eigenvalue {lam:.3e}")
    er, _ = eigensolver.eigenfunction(k)
    v_out.interpolate(er)
    fp.write(v_out, time=k)
print(f"Eigenmodes written to ../output/kform2_2D_{run_tag}.pvd")

if pseudo_export:
    import pseudo_tool
    prefix = f"../output/pseudo_kform2_2D_{run_tag}_npts{npts}"
    pseudo_tool.export_pseudo_spectra_vtk(
        A, M, (-2.0, 30.0, -10.0, 10.0),
        output_prefix=prefix, npts=npts, lams=lams, flip=False,
        B=B if iterative else None,
        pc_solver_parameters=(RIESZ_PC if iterative else None),
        function_space=(V if iterative else None),
    )
