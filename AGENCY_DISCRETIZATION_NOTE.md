# Supervisor query — Agency-channel discretization in the IBF engine

**From:** Radu (LLM substrate work) → Radu (foundation paper author)
**Subject:** A discretization asymmetry between value and agency channels that
matters for small-data regimes
**Status:** Pre-decision; need framework call before patching the engine

---

## TL;DR

The current engine implementation gates the agency channel's `w` update
and `δk` readout by crystallization. The value channel doesn't gate its
`v` update or its `δR` readout that way (same-context transient values
both evolve and contribute). The asymmetry doesn't matter on chess
(crystallization fires quickly under long training with convergent D),
but it matters on LLM closure (small data, persistent contrastive D
signal that never satisfies the convergence_threshold). In our regime,
agency is structurally silenced throughout training, so D5b's positive
A→C emergence came from Boltzmann + value alone, not from
agency-modulated trajectories.

**Question for you:** Is the crystallization gate on agency intended as
a conservative discretization of Postulate 1's "parallel equation," or
is it an under-implementation that should be relaxed to also evolve
transient `w` and include same-context-transient agency in `δk`?

The answer affects:

1. Whether D6 should relax the gate or leave it alone
2. Whether the chess/RRW/CIFAR results need to be re-checked under the
   relaxed dynamics (likely no measurable change in their regimes —
   but worth confirming)
3. Whether the paper's Postulate 1 should be amended to specify the
   discretization, or whether the engine should be amended to match the
   postulate literally read

---

## Where the asymmetry lives (cell 9 of the LLM notebook)

### Value channel — transient dynamics are full citizens

**Create with non-zero v:**

    nc = MemoryCenter(z=z.copy(),
                      v=np.clip(self.p.eta * neg_D, -v_max, v_max),
                      ...)

**Update transient v (faster than crystallized):**

    lr = self.p.eta_cryst if c.is_crystallized() else self.p.eta
    # eta = 0.1 (transient), eta_cryst = 0.005 (crystallized)
    c.v = np.clip(c.v + lr * neg_D * kw, -v_max, v_max)

**Read transient v (same-context only):**

    def _read_gate(self, c):
        if c.context_id == self.current_context_id:
            return 1.0      # same-context: transient OR crystallized both read
        return 1.0 if c.is_crystallized() and c.crucible_verified else 0.0
        # cross-context: requires crystallized AND verified

### Agency channel — transient dynamics are silenced

**Create with w = 0:**

    nc = MemoryCenter(z=z.copy(), mu_eff=self.p.mu_base, context_id=ctx,
                      birth_epoch=self.current_epoch,
                      sigma=self._sigma_agency)
    # no w=... assignment; defaults to 0.0 from dataclass

**Update w only if crystallized:**

    for i, c in enumerate(sc):
        ...
        c.D_history.append(D*kw)
        if c.is_crystallized():       # <— GATE
            dv = c.D_var_rolling()
            tw = np.clip(w_max*(1 - dv/w_dvar_threshold), -w_max, w_max)
            c.w += self.p.eta_k*kw*(tw - c.w)
        # transient: D_history accumulates, w does not move

**Read w only if crystallized:**

    def delta_k(self, z):
        for i, c in enumerate(self.agency_centers):
            if not c.is_crystallized(): continue   # <— GATE
            g = self._read_gate(c)
            if g > 0 and K[i] > self.p.activation_threshold:
                tw += g*c.w*K[i]; sk += g*K[i]

There is no parameter to relax this. `IBFParams` has `eta_k` (the
crystallized-branch learning rate) but no `eta_k_transient`. The
transient agency path simply doesn't exist.

---

## What the paper says

**Postulate 1 (Modification Dynamics):**

> *"A parallel equation governs the responsiveness modification δk_S
> (the agency channel)."*

**Constraint (iv):**

> *"nontrivial interaction generically produces nonzero responsiveness
> modification."*

**Claim 2 (Agency):**

> *"Under the responsiveness modification dynamics, nontrivial
> interaction generically produces spatially nonuniform k_S(z): high
> where local discrepancy signals are consistent (low D-variance), and
> low where alignment is uncertain or contradictory (high D-variance)."*

The literal reading is: any same-context interaction should modify the
local agency, and that modification should be readable. The engine
currently delays both writeability and readability until after
crystallization.

---

## Why it matters here, not in chess

**Chess (paper § 7.2):** 19,054 dissolution events with agency, 30
without. The asymmetry between full and no-agency runs is huge because
crystallization completes for thousands of centers, and post-crystallization
w evolution does what the postulate predicts. The transient phase is
short relative to the run; gating it costs ~nothing.

**LLM closure (D1–D5b):**

- Supervised install of A→B + B→C: 3 epochs × 3 reps = 9 v-updates per
  item. Crystallization needs ≥15 n_updates AND |mean D_history| < 0.025.
  D never converges below 0.025 under contrastive push scheme (target
  centers' D stays ~-0.985, base centers' D stays ~+3).
- Result: 0 new value crystallizations after AB+BC install. 1
  crystallization after 40 epochs of discovery.
- Same gate on agency: 0 to 13 new agency crystallizations across the
  full run, none in the A→C region of 64-D query space (D4 measured
  this directly: 0 of 3,896 ctx-0 value centers have z[:64] within
  1.5·σ_agency of z_64,A).
- Net effect: k_eff stays at k_0 = 5.0 at A→C queries throughout D5b's
  40-epoch discovery run. Agency channel is structurally silent.

D5b's A→C emergence (0.000 → 1.000 on held-out paraphrases) happened
**without** agency engagement — Boltzmann + value channel alone. The
result is paper-grade but tests a degraded version of the
Memory + Agency → Intelligence cascade where the Agency link is
amputated by the gate.

---

## What the relaxation would look like

Three layered changes, all containable to a single domain/run if we
want to test before committing:

**(1) Initialize transient w from creation.** When a new agency center
is nucleated, set initial w from the first D sample analogously to how
value centers set initial v:

    nc.w = np.clip(self.p.eta_k * f(D), -w_max, w_max)

where f(D) maps a single D sample to a tentative w-target (perhaps just
the variance-derived formula with the single sample, or a flat 0 — but
explicitly assigned, not relying on the dataclass default).

**(2) Evolve transient w via the same variance-derived target as
crystallized.** Remove the `if c.is_crystallized()` gate around the w
update, possibly with a separate `eta_k_transient` (analogous to
`eta` vs `eta_cryst` on the value side).

**(3) Include same-context transient agency in δk readout.** Modify
`delta_k` to use `_read_gate` semantics matching `delta_R` — i.e.,
same-context-transient agency contributes, cross-context requires
crystallized + verified.

These three changes are minimal and direct: they bring the agency
channel into structural parallel with the value channel as Postulate 1
specifies.

---

## Risk to existing results

**Chess (paper § 7.2).** Most agency centers crystallize early in
chess training. The relaxation would add transient agency contributions
during the first few epochs of training. Plausible that the early
trajectory shifts slightly (faster k_eff differentiation in low-D-var
regions), but the steady-state would be dominated by the crystallized
contributions as before. **My expectation: chess numbers move
within-noise but the qualitative result (positive BT, agency-vs-no-agency
gap, 19k+ dissolutions) is preserved or strengthened.** Worth re-running
to confirm.

**RRW (paper § 7.1).** Agency is mildly harmful in RRW because
exploration drives corrections into contradiction. Relaxing the gate
would make agency engage *earlier* during training — possibly increasing
the harm slightly. Same qualitative conclusion (regime-dependent agency)
but the No-Agency vs Full-IBF gap might widen. Within the framework's
existing predictions.

**CIFAR (paper § 7.3).** Agency is currently neutral in CIFAR because
the task stream doesn't provide trajectory dependence. Relaxing the
gate shouldn't change this — same null result expected.

**LLM closure (D5b).** Where the relaxation would make a measurable
difference. D6 (proposed below) directly tests this.

---

## Proposed test (D6) before committing to anything

A single D6 cell that:

1. Forks canonical engine fresh.
2. Phase 0: supervised AB+BC install (same as D5b).
3. Phase 1: Boltzmann discovery on A→C with **two parallel runs**:
   - **Run α**: current engine semantics (D5b reproduction)
   - **Run β**: relaxed agency semantics (fixes 1+2+3 above applied
     externally via post-step monkey-patching of `_update_agency` and
     a local `_ungated_delta_k`)
4. Track per-epoch in both runs: AC_hit_rate, AC_k̄ (mean k_eff at AC
   queries), AC test accuracy, AB/BC test accuracy.
5. Decision rule:
   - **β shows k_eff modulation at AC AND emergence is faster/cleaner
     than α**: relaxation validates the full cascade. Recommend
     amending the engine.
   - **β shows k_eff modulation but emergence is no different from α**:
     relaxation is faithful to postulate but doesn't change behavior at
     this scale. Engine choice is empirically harmless.
   - **β shows no k_eff modulation**: even relaxed agency stays
     dormant in the closure region. Deeper coverage issue (σ_agency or
     creation_threshold), not a gating issue.

D6 runs are ~25 min each, so the whole comparison is ~50 min on the
pod, plus a small ablation extension if needed.

---

## What I need from you

1. **Is Reading A or Reading B the intended interpretation of
   Postulate 1?**

2. **If Reading B**: do you want the engine amended (cell 9 patch) or
   should the relaxation stay as an alternate discretization documented
   in the LLM companion paper, with the original engine preserved for
   chess/RRW/CIFAR reproducibility?

3. **If Reading A**: should the paper's Postulate 1 text be tightened
   to make the crystallization gate explicit, or is the gate considered
   a domain-agnostic conservative default that needs no explicit
   statement?

4. **Independent of (1)-(3)**: are you OK with running D6 as a parallel
   comparison (α: status quo, β: relaxed) before deciding? It surfaces
   whether the choice matters empirically on the LLM substrate without
   committing to anything.

---

## Why I'm flagging this rather than just patching

The agency mechanism is one of the four foundational claims of the
paper (Memory, Agency, Self-Correction, Intelligence). If the engine's
agency discretization is a load-bearing detail that wasn't called out
in the paper, the literature derived from this codebase (the LLM
companion, any follow-up work) inherits an ambiguity about which
discretization is canonical. Better to settle this upfront than to
have to retroactively reconcile two papers' implementations.

The fact that D5b's positive result happened *without* agency engaging
is the smoking gun. Either:

- Agency was supposed to engage and didn't (under-implementation in the
  engine — fix and re-run), OR
- Agency genuinely doesn't need to engage in this regime (paper claim
  reads cleanly: "Boltzmann sufficient on this substrate; agency
  remains dormant as predicted by regime-dependence")

Both are valid stories. They lead to different papers. Your call.
