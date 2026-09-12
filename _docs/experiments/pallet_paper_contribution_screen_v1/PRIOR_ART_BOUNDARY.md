# Scope of novelty claims

This is a targeted primary-source check, not an exhaustive novelty review.
Published dataset results are not head-to-head results on our pallets.

| Track | Closest mechanism families checked | Boundary |
|---|---|---|
| A | [Deep Hough Transform for Semantic Line Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/779_ECCV_2020_paper.php); [Deep Hough-Transform Line Priors](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4061_ECCV_2020_paper.php) | Learned features aggregated over line candidates and architectural line priors are prior art. Instance-conditioned local refinement needs its own incremental evidence. |
| C | [Soft Teacher](https://arxiv.org/abs/2106.09018) | Confidence-aware teacher/student detection is established. Freezing geometry while adapting detection is a controlled task-specific intervention, not automatically a novel DA algorithm. |
| D | [Learning to Reweight Examples](https://proceedings.mlr.press/v80/ren18a.html) | Learning which supervision helps is established; this screen differs in supervised signed normal-component gain, not meta-gradient example reweighting. Partial geometry usefulness requires D2>D1, not just teacher quality. |
| B | [Linear-Covariance Loss](https://arxiv.org/abs/2303.11516) | Linearized propagation from keypoint error to pose is prior art. The existing LC-derived implementation is a comparator, never ours/novel. Selecting x/z/yaw components alone does not establish novelty. |
| E | [Learning Using Privileged Information](https://www.jmlr.org/papers/v16/vapnik15b.html) | Extra training-only teacher information and knowledge transfer are prior art. Sensor alignment and actual RGB-only student benefit are separate necessary evidence. |

The D and C distinctions above are our comparison/inference from the cited
mechanisms, not claims made by those papers. No external source code was copied.
CVF HTML fetches were unavailable; author arXiv records supplied the LC and
Soft Teacher primary-source fallback.
