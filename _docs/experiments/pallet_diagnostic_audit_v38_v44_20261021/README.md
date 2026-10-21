# v38–v44: observation, correspondence and selection diagnostics

**Development audits, not an achieved improvement.** This folder publishes selected reports, pseudonymized scalar tables and six user-captured real-photo comparison panels. It adds no training, inference, PnP, new performance evaluation, policy or threshold change. Existing manuscript, method code, checkpoints, original data and prior audit directories are unchanged.

Scientific execution base: [`93c24d1`](https://github.com/CanelE452/pallet-6d-pose/tree/93c24d143fa3901de1fa127b3156e8250d81dba8). Publication parent: `ae99e9f73015d0f98dffe64ae6627575fc86a851`. These are distinct from the local preserved research checkout and from the newer audit versions. Earlier [v19–v37 audit](../pallet_development_audit_v19_v37_20261021/README.md) remains intact.

## What was verified

- **Dimensions (v43):** the 13 inspected real inputs use registered wood W/D/H **80/59/14cm** or plastic **110/130/11cm**. Actual N3 cache input, PnP/D1 geometry and reference dimensions agree. N3 directly receives dimension context; PnP and D1 use dimensioned 3D points; H1 receives resulting geometric candidate features indirectly. Q is a coordinate rotation, not the lengths. Registered nominal/measurement metadata is **not independent remeasurement of each photographed physical pallet**.
- **CAL E84 control (v41/v44):** N0 and original OCPR seed18 have approximately 84° error. PoseFix, SubPix-only and both PF+SubPix anchors also fail. [All stored values](results/E84_all_methods.csv). This case was a posthoc appendix to the original CAL12 audit; that original sample and its proportions were not changed. It also enters v44's fixed all-method-failure control rule. Seed19 differs but still fails; it must not be described as another 84° output.
- **TRAIN8 masks (v42):** existing `>127` decoding matches stored mask area **16/16**, with visible ⊆ amodal **8/8**. Nonzero-background values make a naive nonzero decode wrong. Selected JSONs contain no RLE; exact advanced producer/serialization code was unavailable. Whole-object silhouette does not establish physical face/edge owner. Of 11 partial lines, auxiliary classification is outer support7/internal2/unknown2. Occlusion-cut ambiguity occurs in both failure and success controls. No observation-removal rule is justified.
- **CAL46 comparator success / original OCPR failure (v44):** unique **1/46**, the same frame for all five comparators. N0 **3.909cm/1.855°**, PF+SubPix **4.191cm/1.725°**, OCPR seed18 **7.844cm/1.713°**. Successful N0 remains eligible but ranks37. Stored same-family geometric costs also favor the failed selected pose; this is not evidence that only H1 ignores otherwise unambiguous good observations.
- **PRIMARY192 (v44, posthoc only):** unique **19/192** comparator-any-success/OCPR-failure frames, including **14 with an eligible successful bank candidate / 5 without**. These overlap across comparison methods; method counts must not be summed. Twelve fail only the translation boundary, seven both boundaries; four have R>45°, which does not automatically establish a 90° W/D swap. This is repeatedly exposed development data, not an independent validation set.
- **Night example:** the published pseudonym is identified in [v44 case table](results/v44_cases.csv), case P02. It is in the **bank-good14** group. Successful diagnostic candidate `G10/SWAPPED_WD/endpoint/17` ranks **7**, while `WD0/ORIGINAL_WD/selected` ranks1. OCPR **156.064cm/83.341°**; PFraw **6.223cm/1.752°**; PF+SubPix **1.595cm/2.059°**. Thus the major direction reduction is already present in PFraw, with SubPix reducing remaining translation. The PFraw/PF+SubPix stored hypothesis string is unchanged. Both anchor final point arrays are exactly identical for this frame.

## Keep these contracts separate

| Audit | Population and fixed method | Scope |
|---|---|---|
| v38 | one synthetic VAL v32/v33 case; separate original-OCPR PRIMARY case | stored-model dependency audit and saved-observation cost follow-up; no improvement claim |
| v39 | synthetic TRAIN8, **v5 C2 binary incumbent** | native geometry / whole correspondence audit; diagnostic oracle assignments are not a method |
| v40 | real CAL12, **original OCPR seed18** vs N0 | fixed sampled failures and success controls; not a representative large-rotation cohort |
| v41 | real CAL one E84 posthoc appendix | original CAL12 and its proportions preserved |
| v42 | same synthetic TRAIN8 | renderer mask contract and conditional partial-line source audit |
| v43 | CAL13 and same TRAIN8 | actual retained dimension inputs; no new pose evaluation |
| v44 | real CAL46 main; PRIMARY192 posthoc | original OCPR seed18 vs external methods; other seeds auxiliary |

Success is available pose AND T≤5cm AND R≤5° under the same fixed C2 (I/Y180) reconstructed geometric reference. All comparator pairs have common available/full n46 in CAL and n192 in PRIMARY; missing poses0. A 90° W/D turn was not relabelled correct. Source physical C1/C2 and derived v5 C2 targets are separate contracts. Independent physical appearance C2/C4 labels remain unavailable.

N0 uses N3+SubPix inputs; SubPix-only starts from YOLO points; PalletPoseFix has its own RGB/keypoint model; the combinations apply native SubPix after PF with displacement caps anchored to PF or original YOLO points. They share the image, K, known dimensions and evaluation reference but **not identical upstream inputs/training budgets**. Their improvement is not a pure H1 selector effect. Missing exact intermediate points are marked unmeasured rather than regenerated. Original D1 does not automatically contain external PF final poses.

For v38 read the [completed real-cost follow-up](reports/v38_real_cost_followup_KO.md) after the [original report](reports/v38_KO.md): the earlier report predates the final matrix and solver-stage I0 correction. A/AB names are cost models, distinct from B/I observation roles; AB is not simply A plus an independently additive term. Compare costs only inside a common family and mode, with feasibility recorded separately.

## Reports and evidence

[v38](reports/v38_KO.md) · [v38 real-cost follow-up](reports/v38_real_cost_followup_KO.md) · [v39](reports/v39_KO.md) · [v40](reports/v40_KO.md) · [v41](reports/v41_KO.md) · [v42](reports/v42_KO.md) · [v43](reports/v43_KO.md) · [dimension correction stages](reports/v43_correction_stages_KO.md) · [v44](reports/v44_KO.md)

[Summary CSV](summary.csv) · [v44 method counts](results/v44_method_counts.csv) · [all paired saved rows](results/v44_paired_rows.csv) · [candidate membership](results/v44_membership.csv) · [stored H1 ranks](results/v44_ranks.csv) · [dimensions](results/v43_dimensions.csv) · [manifest and hashes](MANIFEST.json) · [source provenance hashes](PROVENANCE.json)

Representative real panels: [CAL selection damage](figures/cal_selection_damage.png), [E84 all-method failure](figures/cal_E84_all_fail.png), [OCPR success control](figures/cal_ocpr_win_control.png), [PRIMARY bank absence](figures/primary_bank_absence.png), [night PF+SubPix success](figures/primary_night_pf_subpix.png), [N0 selection damage](figures/primary_n0_damage.png). See [figure manifest](FIGURE_MANIFEST.json) and [image rights](IMAGE_RIGHTS.md). The panels use saved pose/point arrays, identical crops and uniform display scaling. Pose-projected corners are not measured keypoints; no generative editing was used.

## Limitations and proposed next diagnostic

The reference is reconstructed geometry, not independent physical GT. Known dimensions do not resolve ambiguous correspondence. Whole-object masks cannot certify edge owner. Auxiliary visual notes are not independent human labels. Success controls can show the same shadow/occlusion/mask patterns as failures; no universal shadow veto is justified. No single frame was used to tune a policy. No independent generalization gain is established.

The v44 next proposal is a small **TRAIN-only fixed 2×2 coordinate-source × W/D-correspondence replay** under a common K/dimensions/solver/budget, conditional on exact stored input/mapping availability. It is not another PF-union fit and does not choose a combination from PRIMARY GT. **It has not been executed.** Missing input or physical-owner contracts remain blockers for conclusions requiring them.

Raw datasets, checkpoints, source images as a full dataset, private ID mappings, personal paths, consultation text, machine details and synthetic photos with unresolved external background/texture rights are omitted. Original private audits remain preserved. This public edition is an evidence package, not a replacement of the manuscript's method or official numerical ledger.

Local verification of this public copy only: `python verify_public_tables.py`. It checks published bytes and saved-row unions, without models, source images or new pose scoring.
