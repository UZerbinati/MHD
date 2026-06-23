"""Iterative shift-and-invert eigensolver with Riesz-map preconditioning.

The 3D k-form scripts ran into MUMPS out-of-memory failures during the
LU factorisation of the shift-and-invert operator (A - sigma M) at
N=64 on the unit box.  This module provides a drop-in iterative
alternative: GMRES on (A - sigma M), preconditioned by the
inner-product matrix B of the function space's natural Sobolev norm —
the operator-preconditioning / Riesz-map approach.

For an SPD inner product B, the spectrum of B^{-1}(A - sigma M) is
mesh-independent (h-robust) when A is the discretisation of an
elliptic / saddle-point operator on the same space.  Hence GMRES
converges in O(1) iterations regardless of mesh size, while B itself
is SPD with a sparse cell-local-or-banded structure that factors
much faster than the indefinite saddle-point matrix A.

What "Riesz form" to use per k-form
-----------------------------------
- DG x RT (kform0):    (v,w)_L2  +  (p,q)_Hdiv = (p,q) + (div p, div q)
- RT x N1curl (kform1): (v,w)_Hdiv + (p,q)_Hcurl
- N1curl x CG (kform2): (v,w)_Hcurl + (psi,phi)_H1
- CG (kform3):          (v,w)_H1

These are the standard mixed-method preconditioning norms; the
discrete spaces match the Sobolev spaces by design.  See
Mardal-Winther 2011 ("Preconditioning discretizations of systems of
partial differential equations") and Schoeberl-Zulehner 2007.

Public API
----------
- ``IterativeEigensolver``: wrapper with the same ``solve / eigenvalue
  / eigenfunction`` surface as ``firedrake.LinearEigensolver`` but
  configurable inner KSP and Pmat.
- ``compute_projections_iterative``: replacement for
  ``pseudo_tool.compute_projections`` that uses the same iterative
  inner solve.
"""
import time
import numpy as np
from firedrake import Function, assemble
from petsc4py import PETSc
import slepc4py
from slepc4py import SLEPc


def orthonormalize_pairs(Vcols, Wcols, tol=1e-10):
    """Modified Gram-Schmidt on the right/left eigenvector pairs, dropping
    pairs where either column is linearly dependent on the previously kept
    ones.  Replaces ``SLEPc.BV.orthogonalize()`` which aborts with a
    "linearly dependent column" breakdown when SLEPc's shift-and-invert
    returns near-duplicate converged Krylov modes.

    Returns ``(V_out, W_out)`` lists of unit PETSc Vecs of equal length.
    """
    V_out, W_out = [], []

    def _mgs(v, basis):
        u = v.copy()
        for _ in range(2):
            for q in basis:
                u.axpy(-q.dot(u), q)
        return u, u.norm()

    for v, w in zip(Vcols, Wcols):
        u_v, nv = _mgs(v, V_out)
        if nv <= tol:
            continue
        u_w, nw = _mgs(w, W_out)
        if nw <= tol:
            continue
        u_v.scale(1.0 / nv)
        u_w.scale(1.0 / nw)
        V_out.append(u_v)
        W_out.append(u_w)
    return V_out, W_out


def assemble_riesz_matrix(form, bcs=None, mat_type="aij"):
    """Assemble a Firedrake UFL form as a PETSc matrix.

    Returned object is a Firedrake ``MatrixBase`` whose ``.petscmat``
    attribute is the PETSc Mat to feed into PETSc/SLEPc.  Wraps
    ``firedrake.assemble`` for clarity at the call site.
    """
    return assemble(form, bcs=bcs, mat_type=mat_type)


_DEFAULT_PC_PARAMETERS = {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


# Monotonic counter so consecutive IterativeEigensolver instances get
# distinct options prefixes — without this, PETSc's global options
# database carries `pc_type=...` from a previous instance into the
# next eps.setFromOptions(), overriding our "none" placeholder PC and
# triggering a (failing) MUMPS factor of the indefinite saddle inside
# eps.setUp().
_OPTIONS_PREFIX_COUNTER = 0


def _next_unique_prefix(stem):
    global _OPTIONS_PREFIX_COUNTER
    _OPTIONS_PREFIX_COUNTER += 1
    return f"{stem}{_OPTIONS_PREFIX_COUNTER}_"


def _apply_options(prefix, parameters):
    """Write a flat solver_parameters dict into PETSc.Options() under
    ``prefix``.  Handles bool flags (True → flag, False → skip), None
    (flag), and stringifies the rest.  Nested fieldsplit options are
    expressed as flat keys with ``_`` separators in the standard PETSc
    convention, e.g. ``fieldsplit_0_pc_type``.
    """
    opts = PETSc.Options()
    for key, value in parameters.items():
        full_key = prefix + key
        if value is False:
            continue
        if value is True or value is None:
            opts.setValue(full_key, None)
        else:
            opts.setValue(full_key, str(value))


def _configure_eps(A_mat, M_mat, B_mat,
                   n_evals,
                   target,
                   which,
                   problem_type,
                   two_sided,
                   ksp_type,
                   ksp_rtol,
                   ksp_atol,
                   ksp_max_it,
                   pc_solver_parameters,
                   options_prefix,
                   comm,
                   monitor,
                   ncv=None,
                   function_space=None,
                   eps_type=None):
    """Internal: build a SLEPc.EPS configured for shift-and-invert with
    iterative inner solve preconditioned by ``B_mat``.

    The Pmat override is the key: SLEPc's STSetUp_Sinvert calls
    ``KSPSetOperators(ksp, T, T)`` where T = A - sigma M, so we must
    redo the override *after* ``eps.setUp()`` and before
    ``eps.solve()``.  KSP's later setUp (triggered inside KSPSolve)
    then uses our B_mat as Pmat to build the PC.

    ``pc_solver_parameters`` is a Firedrake-style flat solver-parameters
    dict (e.g. ``{"pc_type": "fieldsplit", "fieldsplit_0_pc_type": "jacobi",
    ...}``) applied to the inner PC of the shift-and-invert KSP.  When
    ``None``, defaults to LU/MUMPS (preserving the original behaviour).
    """
    if comm is None:
        comm = A_mat.getComm()

    eps = SLEPc.EPS().create(comm=comm)
    # Always use a unique options prefix so back-to-back instances don't
    # cross-contaminate via the global PETSc options database.
    if options_prefix is None:
        options_prefix = _next_unique_prefix("riesz_")
    eps.setOptionsPrefix(options_prefix)
    eps.setOperators(A_mat, M_mat)
    eps.setProblemType(getattr(SLEPc.EPS.ProblemType, problem_type))
    # The default Krylov-Schur EPS hits a PETSc 3.24 complex-build bug
    # in DSDestroy ("MatDenseRestoreSubMatrix" / error 58); plain Arnoldi
    # uses a different DS pathway and is the working workaround.  Caller
    # can override via eps_type=...
    # Arnoldi dodges a Krylov-Schur DS regression in PETSc 3.24's complex
    # build for some operator-state combinations; let callers override
    # the default if needed.
    if eps_type is not None:
        eps.setType(eps_type)
    if ncv is None:
        eps.setDimensions(n_evals)
    else:
        eps.setDimensions(n_evals, ncv=ncv)
    eps.setTarget(target)
    eps.setWhichEigenpairs(getattr(SLEPc.EPS.Which, which))
    if two_sided:
        eps.setTwoSided(True)

    st = eps.getST()
    st.setType("sinvert")
    st.setShift(target)

    ksp = st.getKSP()
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it)
    if monitor:
        ksp.setMonitor(lambda k, it, rn: PETSc.Sys.Print(
            f"    [riesz-ksp] it={it:4d}  ||r||={rn:.3e}", comm=comm))

    # Critical: pre-setUp the PC must be a no-op.  STSetUp_Sinvert
    # eagerly calls KSPSetUp, which would PCSetUp the saddle-point
    # operator T = A - sigma M with whatever PC type is current — and
    # if that's "lu", the factorisation hits the same MUMPS OOM that
    # motivated this whole switch.  So we force pc_type=none, run
    # setUp (which builds T and stores it as both operator and Pmat),
    # then swap Pmat to the Riesz matrix and let setFromOptions install
    # the user-requested PC.
    pc = ksp.getPC()
    pc.setType("none")

    eps.setFromOptions()
    eps.setUp()

    # Override Pmat with the Riesz inner-product matrix.  KSP/PC will
    # (lazily, on the next solve) call PCSetUp on B_mat — SPD and far
    # cheaper than the indefinite T.
    op_mat, _ = ksp.getOperators()
    ksp.setOperators(op_mat, B_mat)

    # Wire the FunctionSpace's DM into the KSP so Python PCs that
    # navigate Firedrake's dmhooks (firedrake.ASMStarPC, PatchPC, GTMG)
    # can recover the function space via
    # ``firedrake.dmhooks.get_function_space``.  The DM handle attached
    # to assembled PETSc matrices is a NULL placeholder, but the
    # function space carries a proper DM at ``V.dm`` with ``__fs_info__``
    # set.  Without this attachment, fieldsplit + ASMStarPC segfaults.
    #
    # We only attach the DM when the requested PC actually needs Firedrake
    # dmhooks (PCPATCH and friends); for plain LU/MUMPS the DM is unused,
    # AND on the PETSc 3.24 complex build attaching it triggers an EPS
    # solve regression ("MatDenseRestoreSubMatrix" / DS error 58) for
    # certain (eps_type, nullspace, _apply_options) combinations.
    pc_params_dict = pc_solver_parameters or {}
    pc_kind = pc_params_dict.get("pc_type", "")
    needs_dm = pc_kind in {"fieldsplit", "patch", "python"}
    if needs_dm and function_space is not None:
        try:
            V_dm = function_space.dm
        except AttributeError:
            V_dm = None
        if V_dm is not None and V_dm.handle:
            ksp.setDM(V_dm)
            ksp.setDMActive(False)  # don't let PETSc re-derive operators from DM

    # Install the user-requested inner PC via the options database under
    # the KSP's prefix, then re-apply.
    params = pc_solver_parameters if pc_solver_parameters is not None \
        else _DEFAULT_PC_PARAMETERS
    ksp_prefix = ksp.getOptionsPrefix() or ""
    _apply_options(ksp_prefix, params)
    pc.setFromOptions()
    ksp.setFromOptions()

    return eps


class IterativeEigensolver:
    """Iterative shift-and-invert eigensolver with Riesz preconditioning.

    Drop-in replacement for ``firedrake.LinearEigensolver`` for the
    3D k-form scripts.  Takes pre-assembled A, M, B PETSc matrices
    plus a Firedrake function space (used to materialise eigenfunctions
    after the solve).

    Parameters
    ----------
    A_mat, M_mat : PETSc Mat
        Bilinear and mass matrices (e.g. ``assemble(...).petscmat``).
    B_mat : PETSc Mat
        Riesz inner-product matrix on the same space; used as the
        preconditioning matrix in the inner GMRES solve.
    function_space : firedrake.FunctionSpaceBase
        Used to allocate Function eigenvectors in ``eigenfunction``.
    n_evals : int
        Number of requested eigenpairs.
    target : float, default 1e-8
        Shift sigma in the shift-and-invert transformation.
    which : str, default ``"TARGET_MAGNITUDE"``
        Member name of ``SLEPc.EPS.Which``.
    problem_type : str, default ``"GNHEP"``
        Member name of ``SLEPc.EPS.ProblemType``.
    two_sided : bool, default False
        If True, enable two-sided eigenvector computation.
    ksp_* : tuning knobs for the outer GMRES on (A - sigma M).
    pc_solver_parameters : dict or None
        Firedrake-style flat solver-parameters dict for the inner PC of
        the shift-and-invert KSP — i.e. the preconditioner of B_mat.
        Keys are PETSc option names (``pc_type``, ``fieldsplit_0_pc_type``,
        ``patch_pc_patch_construct_type``, …).  ``None`` selects
        LU/MUMPS, preserving the original behaviour.

    Use ``-eps_view`` / ``-st_ksp_monitor`` from the command line to
    instrument the run; this class plays nicely with PETSc options.
    """
    def __init__(self, A_mat, M_mat, B_mat, function_space, n_evals,
                 target=1e-8,
                 which="TARGET_MAGNITUDE",
                 problem_type="GNHEP",
                 two_sided=False,
                 ksp_type="fgmres",
                 ksp_rtol=1e-10,
                 ksp_atol=1e-50,
                 ksp_max_it=500,
                 pc_solver_parameters=None,
                 options_prefix=None,
                 monitor=False,
                 ncv=None,
                 comm=None,
                 eps_type=None):
        self._fs = function_space
        self._n_evals = n_evals
        # Only DM-driven PCs (fieldsplit / patch / python) need the
        # dmhooks.add_hooks wrapper around eps.solve(); for plain LU on B
        # we skip it.  See _configure_eps for the matching DM-attachment
        # logic.
        _pc_kind = (pc_solver_parameters or {}).get("pc_type", "")
        self._needs_dm_hooks = _pc_kind in {"fieldsplit", "patch", "python"}
        self.es = _configure_eps(
            A_mat, M_mat, B_mat, n_evals, target, which, problem_type,
            two_sided, ksp_type, ksp_rtol, ksp_atol, ksp_max_it,
            pc_solver_parameters, options_prefix, comm, monitor,
            ncv=ncv, function_space=function_space, eps_type=eps_type,
        )

    def solve(self):
        """Solve the eigenproblem; return the number of converged pairs."""
        # Firedrake-side PCs that go through DM hooks (ASMStarPC,
        # PatchPC, fieldsplit-with-create_field_decomposition) need
        # `__setup_hooks__` to be active on the function-space DM during
        # PCSetUp/KSPSolve; that stack is populated by the ``add_hooks``
        # context manager.  When using LU/MUMPS on B (no DM-driven PCs)
        # this is a no-op AND, on the PETSc 3.24 complex build, the
        # context manager interacts badly with the saved Krylov-Schur DS
        # state -- skipping it dodges the "MatDenseRestoreSubMatrix"
        # cleanup regression.
        from firedrake import dmhooks
        # Only wrap in add_hooks when a DM-driven PC actually needs it.
        # The flag is set in __init__ based on pc_solver_parameters.
        if self._fs is not None and self._needs_dm_hooks:
            with dmhooks.add_hooks(self._fs.dm, self, save=False):
                self.es.solve()
        else:
            self.es.solve()
        nconv = self.es.getConverged()
        if nconv == 0:
            r = self.es.getConvergedReason()
            raise RuntimeError(
                f"IterativeEigensolver did not converge any eigenvalues "
                f"(SLEPc reason {r}).  Increase -st_ksp_max_it / loosen "
                f"-st_ksp_rtol, or check the Riesz form."
            )
        return nconv

    def eigenvalue(self, k):
        """Return the k-th eigenvalue (complex)."""
        return self.es.getEigenvalue(k)

    def eigenfunction(self, k):
        """Return (real_fn, imag_fn) Firedrake Functions for the k-th eigenvector."""
        er = Function(self._fs)
        ei = Function(self._fs)
        with er.dat.vec_wo as vr, ei.dat.vec_wo as vi:
            self.es.getEigenvector(k, vr, vi)
        return er, ei


def compute_projections_iterative(A, M, B,
                                  n_subspace=40,
                                  target=1e-8,
                                  which="TARGET_MAGNITUDE",
                                  shift=None,
                                  ksp_type="fgmres",
                                  ksp_rtol=1e-10,
                                  ksp_max_it=500,
                                  pc_solver_parameters=None,
                                  options_prefix=None,
                                  monitor=False,
                                  function_space=None):
    """Replacement for ``pseudo_tool.compute_projections`` that uses
    iterative + Riesz inner solve for the large eigenproblem.

    ``which`` / ``target`` / ``shift`` have the same meaning as in
    ``pseudo_tool.compute_projections``: ``which="SMALLEST_MAGNITUDE"``
    paired with a small positive ``shift`` lands the projection on the
    large-magnitude bulk of the spectrum (correct for Kikuchi saddle
    pencils, where the bottom is the Lagrange-multiplier null space);
    ``which="TARGET_MAGNITUDE"`` with ``target=t`` lands it near ``t``.

    ``pc_solver_parameters`` is the same flat solver-parameters dict
    accepted by ``IterativeEigensolver``; pass it through from the
    driver to keep the eigensolve and the projection on the same PC.

    Returns ``(Ar_np, Br_np)`` — the small dense projected pencil that
    feeds ``pseudo_tool.compute_residual``.
    """
    tic = time.time()
    eps = _configure_eps(
        A.petscmat, M.petscmat, B.petscmat,
        n_evals=n_subspace,
        target=target,
        which=which,
        problem_type="GNHEP",
        two_sided=True,
        ksp_type=ksp_type,
        ksp_rtol=ksp_rtol,
        ksp_atol=1e-50,
        ksp_max_it=ksp_max_it,
        pc_solver_parameters=pc_solver_parameters,
        options_prefix=options_prefix,
        comm=A.petscmat.getComm(),
        monitor=monitor,
        function_space=function_space,
    )
    if shift is not None:
        eps.getST().setShift(shift)
    if function_space is not None:
        from firedrake import dmhooks
        with dmhooks.add_hooks(function_space.dm, eps, save=False):
            eps.solve()
    else:
        eps.solve()
    nconv = eps.getConverged()
    k = min(n_subspace, nconv)
    if k == 0:
        raise RuntimeError(
            "compute_projections_iterative converged 0 eigenvalues; "
            "the projected pencil is empty.  Increase -st_ksp_max_it "
            "or n_subspace."
        )

    sz = A.petscmat.getSizes()[0]
    Vr = PETSc.Vec().createMPI(sz)
    Vi = PETSc.Vec().createMPI(sz)
    Wr = PETSc.Vec().createMPI(sz)
    Vcols, Wcols = [], []
    for i in range(k):
        eps.getEigenpair(i, Vr, Vi)
        Vcols.append(Vr.copy())
        eps.getLeftEigenvector(i, Wr)
        Wcols.append(Wr.copy())

    V_orth, W_orth = orthonormalize_pairs(Vcols, Wcols)
    k_eff = len(V_orth)
    if k_eff == 0:
        raise RuntimeError(
            "All converged eigenvector pairs were linearly dependent; "
            "cannot build the projected pencil."
        )
    if k_eff < k:
        PETSc.Sys.Print(
            f"Dropped {k - k_eff} linearly dependent eigenpair(s); "
            f"projection rank reduced from {k} to {k_eff}.",
            comm=A.petscmat.getComm(),
        )

    Vbv = SLEPc.BV().create(); Vbv.setSizesFromVec(V_orth[0], k_eff)
    Wbv = SLEPc.BV().create(); Wbv.setSizesFromVec(W_orth[0], k_eff)
    Vbv.setFromOptions(); Wbv.setFromOptions()
    for j, v in enumerate(V_orth):
        Vbv.insertVec(j, v)
    for j, w in enumerate(W_orth):
        Wbv.insertVec(j, w)

    Ar = Vbv.matProject(A.petscmat, Wbv)
    Br = Vbv.matProject(M.petscmat, Wbv)
    Ar.assemble(); Br.assemble()
    Ar_np = np.array(Ar.getDenseArray(), copy=True)
    Br_np = np.array(Br.getDenseArray(), copy=True)
    toc = time.time()
    PETSc.Sys.Print(
        f"Time taken to do Schur decomposition (iterative + Riesz): {toc - tic:.2f}s",
        comm=A.petscmat.getComm(),
    )
    return Ar_np, Br_np
