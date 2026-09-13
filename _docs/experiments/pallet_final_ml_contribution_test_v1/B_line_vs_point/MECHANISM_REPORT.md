# GT-assisted normal/tangent diagnostic

Each GT edge defines its unit tangent/normal. Absolute projected endpoint errors are averaged over the two endpoints before reporting distributions. Shared corners therefore occur in multiple edge roles; these are not independent observations.

| Model | normal median | normal mean | normal P90 | tangent median | tangent mean | tangent P90 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 4.031207 | 14.249778 | 25.193847 | 4.069569 | 13.159059 | 19.457876 |
| L1 | 3.478261 | 13.741185 | 23.962058 | 3.845590 | 12.909681 | 18.639085 |
| L2 | 3.408406 | 13.689195 | 23.987046 | 3.855658 | 12.875790 | 18.503266 |
| L3 | 3.496408 | 13.765009 | 24.131861 | 3.875782 | 12.902383 | 19.175545 |
| P1 | 3.462283 | 13.687038 | 23.995930 | 3.702468 | 12.652746 | 17.315752 |
| P2 | 3.476751 | 13.738355 | 24.553193 | 3.711959 | 12.724408 | 18.050691 |
| P3 | 3.481963 | 13.711863 | 23.777547 | 3.692137 | 12.660125 | 17.233116 |

Seed-mean L−P normal median delta: -0.012641px; tangent median delta: +0.156822px.
All role, difficulty, day/night, material, session and annotation-visibility subgroups are in MECHANISM_NORMAL_TANGENT.json. Difficulty is fixed from R0 frame-mean supervised error (≤5, 5–10, >10px). No subgroup search or new gate is used.
Numeric annotation visibility codes do not independently establish physical observability. No physical occlusion mechanism claim is made. The primary paired real statistic and safety gates govern the verdict; this diagnostic cannot override them.
