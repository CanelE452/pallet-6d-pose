# Track A — frozen development candidate, confirmation NOT_RUN

Candidate: image_line_only, seed1, lambda0.25, temperature1, no cap. New training: 0.
Checkpoint SHA256: `1fc71445c041eb3f14db34d03decfa605b0d7fe0b1f6a6458be8bac0ee436192`.

| Existing arm | kp median px | kp P90 px | translation cm | rotation deg | IoU3D | ADDsym AUC |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 6.6157 | 38.6700 | 7.8969 | 2.2625 | 0.6032 | 0.4285 |
| image_joint | 6.0409 | 37.1011 | 7.5730 | 2.1126 | 0.6301 | 0.4490 |
| geometry_joint | 6.5330 | 38.3499 | 7.7535 | 2.3263 | 0.6043 | 0.4368 |
| image_line_only | 6.0699 | 37.3196 | 7.5485 | 2.1163 | 0.6328 | 0.4490 |

Yaw and per-session/seed results are preserved in A_architecture_confirmation/EXISTING_EVIDENCE.json.
Joint versus line-only remains unresolved; same deployment parameter architecture. Simpler training objective chosen, not a claim of statistical equivalence or faster inference.
Seed1 integrated runtime median17.634ms versus paired R0 median9.152ms; added median8.378ms. These are existing measurements, not a new benchmark.
Existing full-candidate identity/negative-preservation audit is retained. New independent data is not established; source manifests named FINAL are not sufficient.
Verdict: A_NEEDS_NEW_CONFIRMATION_DATA. Existing overall gate failure is unchanged.
