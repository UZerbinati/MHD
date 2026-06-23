"""Per-driver Riesz inner-PC option dicts.

The outer iterative eigensolver in ``riesz_solver`` runs GMRES on
``A - sigma M`` with Pmat overridden to a Riesz inner-product matrix
``B``.  The actual *preconditioner application* is therefore one
``B^{-1}`` solve per outer GMRES iteration.  This module exports
flat solver-parameters dicts (Firedrake style) selecting the right PC
for ``B`` per k-form, given the build constraints (no Hypre, complex
scalars).

Patch construction
------------------
For the H(curl) and H(div) Riesz blocks the substitute for AMS/ADS is
Pavarino-Schoeberl additive Schwarz, implemented in Firedrake as
``ASMStarPC`` (PETSc PCASM under the hood with topological-star
patches).  The patch *entity dimension* per space:

- N1curl in 2D / 3D / on an embedded surface  ->  vertex stars (dim 0)
- RT in 2D                                    ->  vertex stars (dim 0)
- RT in 3D                                    ->  edge stars   (dim 1)

Other Riesz blocks fall back to simpler PCs:

- DG L^2 mass     ->  pc_type=jacobi (exact: element-local block diag)
- CG H^1          ->  pc_type=gamg   (PETSc built-in algebraic MG)

For mixed (saddle-point) problems, ``B`` is block diagonal by
construction, so an additive ``fieldsplit`` over the natural block
decomposition is exact and the per-block PCs are independent.

Use these dicts via::

    from riesz_pcs import KFORM2_3D as PC_PARAMS
    eigensolver = IterativeEigensolver(..., pc_solver_parameters=PC_PARAMS)
"""

# --- atomic block PCs (used INSIDE fieldsplit; carry ksp_type=preonly
# so the fieldsplit sub-block applies the PC exactly once) -------------

_VERTEX_STAR_BLOCK = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMStarPC",
    "pc_star_construct_dim": 0,
}

_EDGE_STAR_BLOCK = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMStarPC",
    "pc_star_construct_dim": 1,
}

_GAMG_BLOCK = {
    "ksp_type": "preonly",
    "pc_type": "gamg",
}

_JACOBI_BLOCK = {
    "ksp_type": "preonly",
    "pc_type": "jacobi",
}


# --- top-level PCs (used as the OUTER inner-PC for single-block
# drivers; no ksp_type — that would override the outer GMRES) ----------

_GAMG_TOPLEVEL = {
    "pc_type": "gamg",
}


def _gamg_wrapped():
    return _wrap_in_inner_ksp(_GAMG_TOPLEVEL)


def _flatten_block(block_idx, block_params):
    """Convert a block-PC dict into ``fieldsplit_<i>_*`` keys."""
    return {f"fieldsplit_{block_idx}_{k}": v for k, v in block_params.items()}


def _wrap_in_inner_ksp(inner_pc_params, ksp_type="cg", ksp_rtol=1e-2,
                       ksp_max_it=50):
    """Wrap an inner-PC dict in a ``pc_type=ksp`` shell so that each
    application of the outer-Riesz PC runs an inner Krylov solve of
    ``B y = x`` to relative tolerance ``ksp_rtol``.

    This is what the Mardal-Winther / Schoeberl-Zulehner Riesz-block
    preconditioning theory actually requires: the outer GMRES on
    (A - sigma M) wants the action of B^{-1}, not just one V-cycle of
    an approximate factorisation.  With only ``preonly`` sub-KSPs the
    additive fieldsplit loses too many iterations on the saddle.

    Result is a flat solver-parameters dict whose top-level PC is a KSP
    shell, with the supplied inner_pc_params fed to the inner PC
    (typically the fieldsplit + PCPATCH composition).
    """
    out = {
        "pc_type": "ksp",
        "ksp_ksp_type": ksp_type,
        "ksp_ksp_rtol": ksp_rtol,
        "ksp_ksp_max_it": ksp_max_it,
        "ksp_ksp_norm_type": "unpreconditioned",
    }
    for k, v in inner_pc_params.items():
        out[f"ksp_{k}"] = v
    return out


def _fieldsplit_additive(block_0_pc, block_1_pc, wrap=True):
    """Compose two atomic block PCs into an additive fieldsplit dict.
    When ``wrap=True`` the result is wrapped in an inner CG so the outer
    GMRES sees a tight B^{-1} approximation."""
    raw = {
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
    }
    raw.update(_flatten_block(0, block_0_pc))
    raw.update(_flatten_block(1, block_1_pc))
    if wrap:
        return _wrap_in_inner_ksp(raw)
    return raw


# --- per-driver dicts ---------------------------------------------------

# Per-driver PCPATCH-fieldsplit dicts.  These are the textbook
# Pavarino-Schoeberl preconditioners for B; they are correct and
# scale to very large meshes, but in practice each PC application
# runs an inner CG with ~20 fieldsplit-additive (vertex-star ASM +
# gamg) iterations, so each outer-GMRES step costs ~10x more wall
# time than LU(B) at moderate N.  Use these when even LU(B) becomes
# too expensive to factor (call site selects via `-pc patch`).

# 2D drivers
KFORM0_2D_PATCH = _fieldsplit_additive(_JACOBI_BLOCK, _VERTEX_STAR_BLOCK)   # DG x RT (2D)
KFORM1_2D_PATCH = _fieldsplit_additive(_VERTEX_STAR_BLOCK, _GAMG_BLOCK)     # N1curl x CG
KFORM2_2D_PATCH = _gamg_wrapped()                                           # CG (single)
HARMONIC_KFORM1_2D_PATCH = KFORM1_2D_PATCH                                  # same as 2D N1curl x CG

# 3D drivers
KFORM0_3D_PATCH = _fieldsplit_additive(_JACOBI_BLOCK, _EDGE_STAR_BLOCK)     # DG x RT (3D)
KFORM1_3D_PATCH = _fieldsplit_additive(_EDGE_STAR_BLOCK, _VERTEX_STAR_BLOCK)  # RT x N1curl
KFORM2_3D_PATCH = _fieldsplit_additive(_VERTEX_STAR_BLOCK, _GAMG_BLOCK)     # N1curl x CG
KFORM3_3D_PATCH = _gamg_wrapped()                                           # CG (single)
KFORM1_MANIFOLD_3D_PATCH = KFORM2_3D_PATCH                                  # embedded surface


# Default Riesz preconditioner: LU/MUMPS on B.  B is SPD with no
# advection coupling, so its factor is several times smaller than the
# saddle LU and fits comfortably at the mesh sizes the paper targets.
# The PCPATCH variants above are reserved for the future regime where
# even LU(B) doesn't fit.
LU_MUMPS = {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# Per-driver default PC dicts: alias the LU_MUMPS dict.  Drivers
# import `KFORM*` and get LU(B) by default; `-pc patch` swaps to the
# `*_PATCH` dict above.
KFORM0_2D = LU_MUMPS
KFORM1_2D = LU_MUMPS
KFORM2_2D = LU_MUMPS
HARMONIC_KFORM1_2D = LU_MUMPS

KFORM0_3D = LU_MUMPS
KFORM1_3D = LU_MUMPS
KFORM2_3D = LU_MUMPS
KFORM3_3D = LU_MUMPS
KFORM1_MANIFOLD_3D = LU_MUMPS


# Mapping driver_name -> (default_dict, patch_dict).  Call sites can
# read `riesz_pcs.RESOLVE[driver][pc_kind]` for a small dispatcher.
RESOLVE = {
    "kform0_2D":           {"lu": KFORM0_2D,           "patch": KFORM0_2D_PATCH},
    "kform1_2D":           {"lu": KFORM1_2D,           "patch": KFORM1_2D_PATCH},
    "kform2_2D":           {"lu": KFORM2_2D,           "patch": KFORM2_2D_PATCH},
    "harmonic_kform1_2D":  {"lu": HARMONIC_KFORM1_2D,  "patch": HARMONIC_KFORM1_2D_PATCH},
    "kform0_3D":           {"lu": KFORM0_3D,           "patch": KFORM0_3D_PATCH},
    "kform1_3D":           {"lu": KFORM1_3D,           "patch": KFORM1_3D_PATCH},
    "kform2_3D":           {"lu": KFORM2_3D,           "patch": KFORM2_3D_PATCH},
    "kform3_3D":           {"lu": KFORM3_3D,           "patch": KFORM3_3D_PATCH},
    "kform1_manifold_3D":  {"lu": KFORM1_MANIFOLD_3D,  "patch": KFORM1_MANIFOLD_3D_PATCH},
}


def select(driver, pc_kind):
    """Look up the PC dict for ``driver`` (e.g. "kform2_3D") and ``pc_kind``
    ("lu" or "patch").  Raises ValueError on unknown combinations so a
    typo in a driver's CLI dispatch surfaces immediately."""
    if driver not in RESOLVE:
        raise ValueError(f"Unknown driver {driver!r}; "
                         f"valid: {sorted(RESOLVE)}")
    if pc_kind not in RESOLVE[driver]:
        raise ValueError(f"Unknown pc_kind {pc_kind!r} for {driver!r}; "
                         f"valid: {sorted(RESOLVE[driver])}")
    return RESOLVE[driver][pc_kind]
