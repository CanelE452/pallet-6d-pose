# Evidence ledger

Every result is classified by **how much its contract was frozen before the result
was seen**, established from git commit ordering, not from what a document claims
about itself.

Method: each lock's commit time was compared with the commit time of the result it
governs. All eleven core lock files were additionally checked for post-hoc editing —
worktree hash against committed blob hash. **None has been modified since it was
committed.**

## Tier definitions

```text
TIER A   the method and training contract were frozen, in an earlier commit,
         before any result on the evaluation population was observed
TIER B   designed or tuned after PAPER_EVAL results had been seen;
         usable for mechanism analysis, never as confirmation
TIER C   reference or upper bound; not a controlled comparison
```

## Tier A — confirmatory

```text
result                lock commit              result commit            lead
──────────────────────────────────────────────────────────────────────────────
R0                    15f0cb5  09-01 18:28:54  89129c2  09-01 22:39:59  4h11m
R0-CONT               c3a2581  09-01 18:53:39  89129c2  09-01 22:39:59  3h46m
V1 R1 naive           c3a2581 + dceb83a        89129c2  09-01 22:39:59  3h46m
V1 R2 confidence      c3a2581 + dceb83a        89129c2  09-01 22:39:59  3h46m
V1 R3 + reprojection  c3a2581 + dceb83a        89129c2  09-01 22:39:59  3h46m
V1 R4 + removal       c3a2581 + dceb83a        89129c2  09-01 22:39:59  3h46m
V1 R5 full filter     c3a2581 + dceb83a        89129c2  09-01 22:39:59  3h46m
V1 A2 matched         c3a2581  09-01 18:53:39  cc924c0  09-01 23:54:32  5h01m
V1 A12 sensitivity    c3a2581  09-01 18:53:39  9b412c4  09-02 08:00:52  13h07m
DOPE M1 baseline row  15f0cb5  09-01 18:28:54  8e6ccbc  09-02 07:48:59  13h20m
```

The V1 block is the strongest evidence in the study. Three separate locks —
`PRE_RESULT_LOCK`, `ADAPTATION_POOL_LOCK` + `PSEUDOLABEL_FILTER_LOCK`, and
`SELFTRAIN_EXPOSURE_LOCK` — were committed hours before the results, name every arm
in `arms[]` in advance, pin the initialisation checkpoint by sha256, and carry
`GT_USED_FOR_SELECTION = false` and `paper_eval_not_consulted = true`.

Weaker A, flagged: **V1 A8** (day/night appendix) was registered in `EXPERIMENTS.md`
rather than in a lock file. Treat it as A with a caveat, not as equal to the arms
above.

## Tier B — development and post-hoc diagnostic

```text
result                       why it is B
────────────────────────────────────────────────────────────────────────────────
V2 (V2A-V2D)                 the diagnoses it was designed from were committed
                             47-60 minutes before its method lock; the lock itself
                             marks its thresholds DEV-INFORMED, NOT PREREGISTERED
V3 (V3A, V3B)                lock sets dev_informed = true and names PAPER_EVAL 319
                             as its development population
V4                           lock and result are in the SAME commit (0c36084);
                             ordering cannot be established from git at all
V5                           lock leads the result by 12 minutes, but the lock
                             itself records the population as already consumed by
                             V1-V4
FILTER_SEPARABILITY          no lock, no purpose file; the measurement script is in
                             the same commit as its result
FAST teacher A, B, C         no method lock; contract and results in one commit
STRONG teacher audit         purpose, lock and result all in one commit, 89 seconds
                             apart by file timestamp
V1 R6_CONF_FLIP              not in the exposure lock's arms[]; first appears in
                             its own result commit, after R1-R5 were seen
V1 B_CONF_* (3 arms)         not in the lock; created 14 hours after R1-R5 results
V1 P43/P44 replicates        the commit message states they replace an earlier
                             sweep — a design change made after seeing results
```

**Three of these must never be described as pre-registered**, because git cannot
order them at all:

```text
V4                lock and result in commit 0c36084
STRONG teacher    purpose, lock and result in commit a9d4f32
FAST teacher      freeze and results in commit c23959a
```

File modification times suggest the intended order in each case, but a file
timestamp is not provenance. They are reported as diagnostics and nothing more.

## Tier C — reference and upper bound

```text
DOPE same-data backbone control    RESULT.json predates the evaluator contract by
                                   7h18m, so it sits outside the locked evaluation;
                                   the separately re-scored M1 row is the Tier A one
Real-FT (REALFT_A/B/LV1V2)         trained with real supervision; its own commit
                                   message states it is not a controlled comparison
```

Real-FT is an upper bound. It never appears in the same column block as the
unlabeled-adaptation arms.

## Unverified

```text
STRONG teacher T1/C1/S1 gates   the audit cites _docs/paper/STRONG_TEACHER_V1_PROTOCOL.md
                                as the gate definition. That file does not exist
                                anywhere in the repository or the working tree, so
                                the gates cannot be shown to have been pre-registered.
REAL_FT_V1                      see the correction below
```

REAL_FT_V1 was designed and audited before training, but its method lock was
committed later in the paper-finalization commit `818bd3e`, after the track had
already been stopped. The committed lock documents the intended design but cannot
establish preregistration. No training result exists. **It is not Tier A and must
never be promoted there.**

An earlier draft of this ledger said the lock was "never committed". That was
wrong: it is in `818bd3e`. What it cannot do is order itself against a result,
because there is no result.

## Two defects that affect citation

```text
PRE_RESULT_LOCK self-reference
  the lock records "commit": "0a872a19..." which is not an ancestor of the current
  history. 0a872a1 and 15f0cb5 share a parent and a timestamp and differ only by the
  lock's own SHA line — a commit-then-amend artefact, not back-dating.
  Do not cite 0a872a19 in the paper; a reader cannot resolve it. Cite 15f0cb5.

FILTER_SEPARABILITY has no purpose file
  it is a post-hoc measurement over a population already consumed by V1-V4.
  It supports "the signals are not random"; it cannot support any claim about a
  filter's downstream effect.
```

## What this ledger permits and forbids

```text
permitted   main tables built from Tier A arms
permitted   Tier B results in Discussion and Appendix, labelled as development
permitted   Tier C rows labelled as reference, with their asymmetries footnoted
────────────────────────────────────────────────────────────────────────────────
forbidden   calling any Tier B result an independent confirmation
forbidden   calling any number on PAPER_EVAL a held-out result
            (population_contract.role = DEV, held_out_final = false)
forbidden   presenting V3-B as the proposed method
forbidden   describing V4, STRONG teacher, or FAST teacher as pre-registered
forbidden   promoting R6_CONF_FLIP or the B_CONF_* arms into the main comparison
```

## The population itself

```text
PAPER_EVAL   319 positive frames, 2,689 real negative frames
             population_contract.role = DEV
             held_out_final = false
             consumed as a development population by V2, V3, V4, V5,
             FILTER_SEPARABILITY, and all three teacher probes
```

There is no untouched confirmation population in this study. That is a limitation,
it is stated in the Limitations section, and it is the reason the paper's positive
claims are restricted to Tier A arms.
