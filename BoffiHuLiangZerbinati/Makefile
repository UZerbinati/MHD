# Makefile for the FEEC advection-diffusion paper experiments.
#
# Drivers used by the paper (each writes to experiments/output/):
#   2D/kform0_2D.py          k=0 full-Hodge box driver (DG x RT)
#   2D/kform2_2D.py          k=2 primal CG box driver
#   2D/pseudo_kform1_2D.py   k=1 Kikuchi flat-torus / flat-cylinder driver
#   2D/harmonic_kform1_2D.py harmonic 1-forms on the embedded torus
#
# Figure renderers (each reads cached output, writes to figures/):
#   figures/render_pseudospectra_2D.py        -> kform{0,2}_2D plates
#   figures/render_pseudospectra_kform1_2D.py -> kform1 flat-torus/cylinder plates
#   figures/render_harmonic_3D.py             -> harmonic torus plates
#
# Top-level targets (see `make help`):
#   make data       -- run every experiment needed for the paper
#   make figures    -- render every figure from cached experiment data
#   make paper      -- data + figures (one-shot rebuild from scratch)
#   make clean      -- remove experiments/output
#
# All commands respect Make's -j flag for parallel execution.

SHELL    := /bin/bash
PYTHON   ?= /home/umbertoz/Devel/fd/release-complex/bin/python
EXP      := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT     := $(abspath $(EXP)/..)
FIG_DIR  := $(ROOT)/figures
PYFLAGS  ?= -W ignore
LOG_DIR  := $(EXP)/output/logs

# Tunable mesh / solver parameters
N_2D     ?= 256       # box mesh (kform0_2D, kform2_2D)
N_FLAT   ?= 128       # periodic mesh (pseudo_kform1_2D)
N_HARM   ?= 128       # embedded surface mesh (harmonic_kform1_2D)
ORDER    ?= 1
NEIG     ?= 20
NPTS     ?= 50000

# Wind menus, one per figure
WINDS_PAPER       ?= 0 1 4 5   # fig:pseudospectra, fig:pseudospectra:kform1:flat
WINDS_HARM        ?= 0 1       # fig:harmonic-torus  (all Re in RES_HARM)
WINDS_HARM_EXTRA  ?= 3 4       # fig:harmonic-torus-extra (only at Re=100)
RES_HARM          ?= 0.1 1 10 100
PDIRS_FLAT        ?= both x

# Per-job address-space cap (GiB).  Each recipe runs `ulimit -v` so the
# Python process is killed if it grows past this.  Aggregate ceiling
# under `make -j N` is MEM_GB * N, so divide MEM_GB by N if budgeted.
MEM_GB   ?= 80
MEM_KB    = $(shell echo $$(( $(MEM_GB) * 1024 * 1024 )))
ULIMIT    = ulimit -v $(MEM_KB)

# Inner KSP selector for the eigensolves
#   direct    -> LU/MUMPS on the saddle  (default; reproduces the paper)
#   iterative -> GMRES + Riesz with the per-driver PCs in riesz_pcs.py
KSP_2D   ?= direct
KSP_HARM ?= direct
ITER_BOOL = $(if $(filter iterative,$(1)),1,0)
INNER_PC ?= lu

# Pseudo-spectra SLEPc flag bundle for the k=1 flat-torus / cylinder driver
# (mirrors the validated bundle from pseudo_maxwell_schur.py).
PSEUDO_KFORM1_2D_FLAGS ?= -pseudo_svd_type trlanczos -pseudo_st_shift 0.1 \
                          -pseudo_svd_max_it 10000   -pseudo_st_type sinvert \
                          -pseudo_st_pc_type qr      -pseudo_eps_target 0.1

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------

.PHONY: help
help:
	@echo "Top-level targets:"
	@echo "  data             -- run every experiment needed for the paper"
	@echo "  figures          -- render every figure from cached experiment data"
	@echo "  paper            -- data + figures (one-shot rebuild)"
	@echo "  pseudo-2D        -- k=0 + k=2 pseudospectra (winds=$(WINDS_PAPER) at Re=1)"
	@echo "  pseudo-kform1-2D -- k=1 pseudospectra (winds=$(WINDS_PAPER) x pdir=$(PDIRS_FLAT))"
	@echo "  harmonic         -- harmonic 1-forms on embedded torus"
	@echo "                       (winds=$(WINDS_HARM) at every Re in $(RES_HARM),"
	@echo "                        plus winds=$(WINDS_HARM_EXTRA) at Re=100)"
	@echo "  abc-convergence  -- 3D ABC-flow convergence sweep (N in $(N_ABC_LIST) at Re=$(Re_ABC));"
	@echo "                       writes experiments/output/abc_convergence.csv"
	@echo "  clean            -- rm -rf experiments/output"
	@echo ""
	@echo "Variables (override on the command line):"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  N_2D=$(N_2D)  N_FLAT=$(N_FLAT)  N_HARM=$(N_HARM)"
	@echo "  ORDER=$(ORDER)  NEIG=$(NEIG)  NPTS=$(NPTS)"
	@echo "  WINDS_PAPER='$(WINDS_PAPER)'  WINDS_HARM='$(WINDS_HARM)'"
	@echo "  WINDS_HARM_EXTRA='$(WINDS_HARM_EXTRA)'"
	@echo "  PDIRS_FLAT='$(PDIRS_FLAT)'  RES_HARM='$(RES_HARM)'"
	@echo "  MEM_GB=$(MEM_GB)  KSP_2D=$(KSP_2D)  KSP_HARM=$(KSP_HARM)  INNER_PC=$(INNER_PC)"
	@echo ""
	@echo "Run sub-targets standalone (e.g. 'make pseudo-2D') and parallelise"
	@echo "with 'make -j N'; set MEM_GB = (total budget) / N to keep the sum bounded."

# ---------------------------------------------------------------------------
# k=0 / k=2 pseudospectra on [0, pi]^2  (DG x RT and CG)
# Each driver runs with -pseudo_export 1 so it writes both the eigenmode
# PVDs and the pseudo-spectra VTK pair consumed by render_pseudospectra_2D.py.
# ---------------------------------------------------------------------------

DRV_PSEUDO_2D := kform0_2D kform2_2D

define MAKE_PSEUDO_2D
.PHONY: pseudo-$(1)-w$(2)
pseudo-$(1)-w$(2):
	@echo "[pseudo] $(1) wind=$(2) Re=1.0  (mem cap $(MEM_GB)GB, ksp=$(KSP_2D), pc=$(INNER_PC))"
	@mkdir -p $(LOG_DIR)
	set -o pipefail; cd $(EXP)/2D && $(ULIMIT) && $(PYTHON) $(PYFLAGS) $(1).py \
	    -wind $(2) -Re 1.0 -N $(N_2D) -order $(ORDER) -neig $(NEIG) \
	    -pseudo_export 1 -npts $(NPTS) \
	    -iterative $(call ITER_BOOL,$(KSP_2D)) -pc $(INNER_PC) \
	    2>&1 | tee $(LOG_DIR)/$$@.log
endef

$(foreach d,$(DRV_PSEUDO_2D),$(foreach w,$(WINDS_PAPER),\
    $(eval $(call MAKE_PSEUDO_2D,$(d),$(w)))))

ALL_PSEUDO_2D := $(foreach d,$(DRV_PSEUDO_2D),$(foreach w,$(WINDS_PAPER),\
    pseudo-$(d)-w$(w)))

.PHONY: pseudo-2D
pseudo-2D: $(ALL_PSEUDO_2D)

# ---------------------------------------------------------------------------
# k=1 flat-torus / flat-cylinder pseudospectra  (Kikuchi N1curl x CG)
# pdir=both -> flat torus,  pdir=x -> flat cylinder
# ---------------------------------------------------------------------------

define MAKE_PSEUDO_KFORM1_2D
.PHONY: pseudo-kform1-2D-w$(1)-pdir$(2)
pseudo-kform1-2D-w$(1)-pdir$(2):
	@echo "[pseudo-kform1-2D] wind=$(1) pdir=$(2) Re=1.0  (mem cap $(MEM_GB)GB)"
	@mkdir -p $(LOG_DIR)
	set -o pipefail; cd $(EXP)/2D && $(ULIMIT) && $(PYTHON) $(PYFLAGS) pseudo_kform1_2D.py \
	    -Re 1.0 -wind $(1) -periodic_direction $(2) \
	    -N $(N_FLAT) -neig $(NEIG) -npts $(NPTS) -invert_stencil \
	    $(PSEUDO_KFORM1_2D_FLAGS) \
	    2>&1 | tee $(LOG_DIR)/$$@.log
endef

$(foreach w,$(WINDS_PAPER),$(foreach p,$(PDIRS_FLAT),\
    $(eval $(call MAKE_PSEUDO_KFORM1_2D,$(w),$(p)))))

ALL_PSEUDO_KFORM1_2D := $(foreach w,$(WINDS_PAPER),$(foreach p,$(PDIRS_FLAT),\
    pseudo-kform1-2D-w$(w)-pdir$(p)))

.PHONY: pseudo-kform1-2D
pseudo-kform1-2D: $(ALL_PSEUDO_KFORM1_2D)

# ---------------------------------------------------------------------------
# Harmonic 1-forms on the embedded torus  (harmonic_kform1_2D.py -geo torus)
# WINDS_HARM       -> every Re in RES_HARM   (fig:harmonic-torus)
# WINDS_HARM_EXTRA -> only at Re=100         (fig:harmonic-torus-extra)
# ---------------------------------------------------------------------------

define MAKE_HARMONIC
.PHONY: harmonic-w$(1)-Re$(2)
harmonic-w$(1)-Re$(2):
	@echo "[harmonic] geo=torus wind=$(1) Re=$(2)  (mem cap $(MEM_GB)GB, ksp=$(KSP_HARM))"
	@mkdir -p $(LOG_DIR)
	set -o pipefail; cd $(EXP)/2D && $(ULIMIT) && $(PYTHON) $(PYFLAGS) harmonic_kform1_2D.py \
	    -geo torus -wind $(1) -Re $(2) \
	    -N $(N_HARM) -order $(ORDER) -neig $(NEIG) \
	    -iterative $(call ITER_BOOL,$(KSP_HARM)) \
	    2>&1 | tee $(LOG_DIR)/$$@.log
endef

$(foreach w,$(WINDS_HARM),$(foreach r,$(RES_HARM),\
    $(eval $(call MAKE_HARMONIC,$(w),$(r)))))
$(foreach w,$(WINDS_HARM_EXTRA),\
    $(eval $(call MAKE_HARMONIC,$(w),100)))

ALL_HARMONIC := $(foreach w,$(WINDS_HARM),$(foreach r,$(RES_HARM),harmonic-w$(w)-Re$(r))) \
                $(foreach w,$(WINDS_HARM_EXTRA),harmonic-w$(w)-Re100)

.PHONY: harmonic
harmonic: $(ALL_HARMONIC)

# ---------------------------------------------------------------------------
# ABC-flow Maxwell half-Hodge convergence test on a 3D periodic cube
# (3D/kform1_abc_3D.py).  Beltrami symmetry gives the closed-form
# eigenpair (lambda*, B*) = (1/Rm, u); the script targets it on a sequence
# of refined meshes and appends one CSV row per N to abc_convergence.csv.
# ---------------------------------------------------------------------------

N_ABC_LIST ?= 4 10 16 22 28 32 40 48 56 64
Re_ABC     ?= 1
NEIG_ABC   ?= 4
# Solver path for the ABC sweep:
#   direct    -> LU/MUMPS on the indefinite saddle (cheap at small N,
#                fills aggressively at N >= 32 in 3D)
#   iterative -> GMRES on (A - sigma M) preconditioned by LU(B), where
#                B is the H(curl) x H^1 Riesz map.  Only B is factored,
#                and LU(B) has far less fill than LU(A - sigma M).
# Solver path for the ABC sweep:
#   direct    -> LU/MUMPS on the indefinite saddle.
#   iterative -> GMRES on (A - sigma M) preconditioned by LU(B).  This
#                hits a PETSc 3.24 complex-build SLEPc bug in our state
#                (error 58 inside DSDestroy / MatDenseRestoreSubMatrix);
#                the bug only fires for certain operator/options
#                combinations, none of which we've been able to isolate
#                inside the IterativeEigensolver flow.  Reproducer kept
#                in experiments/3D/kform1_abc_3D.py's iterative branch
#                so the fix can be tested when SLEPc is updated.
KSP_ABC    ?= direct
ABC_CSV     = $(EXP)/output/abc_convergence.csv

define MAKE_ABC
.PHONY: abc-N$(1)
abc-N$(1):
	@echo "[abc] N=$(1) Re=$(Re_ABC)  (mem cap $(MEM_GB)GB, ksp=$(KSP_ABC))"
	@mkdir -p $(LOG_DIR)
	set -o pipefail; cd $(EXP)/3D && $(ULIMIT) && $(PYTHON) $(PYFLAGS) kform1_abc_3D.py \
	    -N $(1) -Re $(Re_ABC) -order $(ORDER) -neig $(NEIG_ABC) \
	    -iterative $(call ITER_BOOL,$(KSP_ABC)) \
	    2>&1 | tee $(LOG_DIR)/$$@.log
endef

$(foreach n,$(N_ABC_LIST),$(eval $(call MAKE_ABC,$(n))))

# Aggregate: wipe the CSV first, then run leaves sequentially via a
# shell loop so the CSV ends up in increasing-N order even under `-j N`.
.PHONY: abc-convergence
abc-convergence:
	@mkdir -p $(EXP)/output
	@rm -f $(ABC_CSV)
	@echo "[abc-convergence] sweep N in $(N_ABC_LIST) at Re=$(Re_ABC)"
	@for n in $(N_ABC_LIST); do \
	   $(MAKE) --no-print-directory abc-N$$n || exit 1; \
	 done
	@echo ""
	@echo "Convergence CSV: $(ABC_CSV)"
	@cat $(ABC_CSV) 2>/dev/null || true

# ---------------------------------------------------------------------------
# Figure renderers (read cached experiment output, write to figures/)
# Renderers are independent and can run in parallel under `make -j`.
# ---------------------------------------------------------------------------

.PHONY: figures
figures: figures-pseudospectra-2D figures-pseudospectra-kform1-2D figures-harmonic figures-abc

.PHONY: figures-pseudospectra-2D
figures-pseudospectra-2D:
	@echo "[figures] render_pseudospectra_2D.py  (kform0 + kform2 plates)"
	cd $(FIG_DIR) && $(PYTHON) render_pseudospectra_2D.py

.PHONY: figures-pseudospectra-kform1-2D
figures-pseudospectra-kform1-2D:
	@echo "[figures] render_pseudospectra_kform1_2D.py  (flat torus + flat cylinder)"
	cd $(FIG_DIR) && $(PYTHON) render_pseudospectra_kform1_2D.py

.PHONY: figures-harmonic
figures-harmonic:
	@echo "[figures] render_harmonic_3D.py  (harmonic 1-forms on the embedded torus)"
	cd $(FIG_DIR) && $(PYTHON) render_harmonic_3D.py

.PHONY: figures-abc
figures-abc:
	@echo "[figures] render_abc_eigenfunction.py  (ABC eigenfunction on the 3D cube)"
	cd $(FIG_DIR) && $(PYTHON) render_abc_eigenfunction.py

# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

.PHONY: data
data: pseudo-2D pseudo-kform1-2D harmonic

# Use recursive make so `data` finishes before `figures` even under `-j N`.
.PHONY: paper
paper:
	$(MAKE) data
	$(MAKE) figures

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

.PHONY: clean
clean:
	rm -rf $(EXP)/output
