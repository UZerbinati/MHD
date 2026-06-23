from firedrake import *
import slepc4py
import petsc4py
import sys
petsc4py.init(sys.argv)
slepc4py.init(sys.argv)
from slepc4py import SLEPc
from petsc4py import PETSc
import time
import os
import numpy as np
from matplotlib.tri import Triangulation
from tqdm import tqdm


def compute_residual(z, A, B, nsvd=3):
        svd = SLEPc.SVD()
        svd.create()
        Bs = B.duplicate(copy=True)
        Bs.scale(-z)
        C = A.duplicate
        C = A+Bs
        svd.setOperators(C, B)
        svd.setDimensions(nsvd)
        svd.setWhichSingularTriplets(SLEPc.SVD.Which.SMALLEST)  # focus on smallest singular values
        svd.setOptionsPrefix("pseudo_")
        svd.setFromOptions()
        svd.solve()

        # Get the number of converged singular values
        nconv = svd.getConverged()
        if nconv > 0:
        # Get the smallest singular value
            sval = svd.getValue(0)
            return sval
        else:
            raise RuntimeError("SVD did not converge")
            return 0.0
def compute_projections(A, M, n_subspace=40, opts=None):
    tic = time.time()
    eps = SLEPc.EPS().create()
    eps.setOperators(A.petscmat, M.petscmat)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setDimensions(n_subspace)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
    eps.setTwoSided(True) #Two sided converge with correct order for NHP (observation by Yuji!)
    eps.setOptionsPrefix("pseudo_")
    eps.setFromOptions()
    eps.solve()
    nconv = eps.getConverged()
    k = min(n_subspace, nconv)
    # Collect the first k right/left eigenvectors into PETSc Vecs
    Vr, Vi = PETSc.Vec().createMPI(A.petscmat.getSizes()[0]), PETSc.Vec().createMPI(A.petscmat.getSizes()[0])
    Wr, Wi = PETSc.Vec().createMPI(A.petscmat.getSizes()[0]), PETSc.Vec().createMPI(A.petscmat.getSizes()[0])

    Vcols, Wcols = [], []
    for i in range(k):
        eps.getEigenpair(i, Vr, Vi)       # right eigenvector (real/imag parts if complex build uses split)
        Vcols.append(Vr.copy())
        eps.getLeftEigenvector(i, Wr)     # left eigenvector i
        Wcols.append(Wr.copy())
    #Project the pencil to the subspace
    Vbv = SLEPc.BV().create(); Vbv.setSizesFromVec(Vcols[-1], n_subspace)
    Wbv = SLEPc.BV().create(); Wbv.setSizesFromVec(Wcols[-1], n_subspace)
    Vbv.setFromOptions(); Wbv.setFromOptions()

    for j,v in enumerate(Vcols): Vbv.insertVec(j, v)
    for j,w in enumerate(Wcols): Wbv.insertVec(j, w)
    try:
        Vbv.orthogonalize()    # optional but helpful
        Wbv.orthogonalize()
    except (SystemError, PETSc.Error):
        pass

    # Ar = W^H A V,  Br = W^H B V
    Ar = PETSc.Mat().createDense([k,k]); Ar.setUp()
    Br = PETSc.Mat().createDense([k,k]); Br.setUp()
    Ar = Vbv.matProject(A.petscmat, Wbv)   # oblique projection Y^H*A*X
    Br = Vbv.matProject(M.petscmat, Wbv)   # oblique projection Y^H*A*X
    Ar.assemble(); Br.assemble()
    toc = time.time()
    print(f"Time taken to do Schur decomposition: {toc-tic}")
    return Ar, Br


def _write_triangulated_vtk(path, x, y, triangles, residual):
    """Write the residual sampled at scattered (x, y) points with a precomputed
    triangulation as a legacy VTK UNSTRUCTURED_GRID of linear triangles."""
    n = x.size
    n_tri = triangles.shape[0]
    eps = np.finfo(float).tiny
    log_residual = np.log10(np.maximum(residual, eps))
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Pseudo-spectra residual sigma_min(A - z M)\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n} float\n")
        for i in range(n):
            f.write(f"{x[i]:.8e} {y[i]:.8e} 0.0\n")
        f.write(f"CELLS {n_tri} {4*n_tri}\n")
        for t in triangles:
            f.write(f"3 {int(t[0])} {int(t[1])} {int(t[2])}\n")
        f.write(f"CELL_TYPES {n_tri}\n")
        for _ in range(n_tri):
            f.write("5\n")  # VTK_TRIANGLE
        f.write(f"POINT_DATA {n}\n")
        f.write("SCALARS residual float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in residual:
            f.write(f"{v:.8e}\n")
        f.write("SCALARS log10_residual float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in log_residual:
            f.write(f"{v:.8e}\n")


def _write_eigenvalues_vtk(path, lams):
    """Write a list of complex eigenvalues as legacy VTK POLYDATA point cloud."""
    lams = list(lams) if lams is not None else []
    n = len(lams)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Eigenvalues in the complex plane\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for lam in lams:
            f.write(f"{float(lam.real):.8e} {float(lam.imag):.8e} 0.0\n")
        if n > 0:
            f.write(f"VERTICES {n} {2*n}\n")
            for i in range(n):
                f.write(f"1 {i}\n")
            f.write(f"POINT_DATA {n}\n")
            f.write("SCALARS real_part float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for lam in lams:
                f.write(f"{float(lam.real):.8e}\n")
            f.write("SCALARS imag_part float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for lam in lams:
                f.write(f"{float(lam.imag):.8e}\n")
            f.write("SCALARS magnitude float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for lam in lams:
                f.write(f"{abs(complex(lam)):.8e}\n")


def export_pseudo_spectra_vtk(A, M, rect, output_prefix, npts=2000, lams=None, flip=False, opts=None,
                              B=None, pc_solver_parameters=None, function_space=None):
    """Sample the pseudo-spectra residual at random points in `rect`, build a
    Delaunay triangulation of the samples, and write both the triangulated
    residual field and the (optional) eigenvalue cloud as legacy VTK files
    openable in ParaView.

    Parameters
    ----------
    A, M : firedrake assembled matrices.
    rect : (minX, maxX, minY, maxY) sampling window in the complex plane.
    output_prefix : path prefix; "<prefix>_pseudospectra.vtk" and
        "<prefix>_eigenvalues.vtk" will be written (parent directory is
        created if missing).
    npts : number of random samples in the rectangle.
    lams : iterable of complex eigenvalues to embed in the eigenvalue file.
    flip : if True, sample at -z and mirror coordinates (matches the previous
        plotting convention).
    """
    print("Computing projections...")
    Ar, Br = compute_projections(A, M, opts=opts)

    # `BV.matProject` returns a SEQDENSE replicated on every rank, so the
    # cheap residual loop, random sampling, and VTK writes all run on rank 0.
    if A.petscmat.getComm().getRank() != 0:
        return None, None

    minX, maxX, minY, maxY = rect
    npts = max(int(npts), 3)
    x = np.random.uniform(minX, maxX, npts)
    y = np.random.uniform(minY, maxY, npts)
    Z = (x + 1j * y)
    if flip:
        Z = -Z

    print(f"Evaluating residual at {npts} random points...")
    residual = np.empty(npts)
    tic = time.time()
    with tqdm(total=npts) as pbar:
        for k in range(npts):
            residual[k] = compute_residual(Z[k], Ar, Br)
            pbar.update(1)
    toc = time.time()
    print(f"Time taken to evaluate pseudo-spectra: {toc-tic:.2f} s")

    if flip:
        x_out, y_out = -x, -y
    else:
        x_out, y_out = x, y
    tri = Triangulation(x_out, y_out)

    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    field_path = f"{output_prefix}_pseudospectra.vtk"
    eig_path = f"{output_prefix}_eigenvalues.vtk"
    _write_triangulated_vtk(field_path, x_out, y_out, tri.triangles, residual)
    _write_eigenvalues_vtk(eig_path, lams)
    print(f"Wrote pseudo-spectra field to {field_path}")
    print(f"Wrote eigenvalues       to {eig_path}")
    return field_path, eig_path
