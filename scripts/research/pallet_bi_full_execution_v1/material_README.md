S3/S4 registered structured proxy experiment
===========================================

This new campaign uses the existing textured photogrammetry/processed scan OBJ
and existing pallet USD geometry. A known model name, or known registered
physical dimensions for real objects, selects a single fixed proxy. The missing
two GLB assets use the fixed structured wood USD proxy. All instances retain the
PROXY label: exact real CAD identity, signed physical axes, and independent
physical owner labels are unverified. Mesh edges are not cuboid corner IDs.

The shared automatic support archive uses welded mesh crease/boundary edges,
current-N0 non-MSAA z-buffer visibility, whole native RGB gradients, and the
existing automatic tight/full masks. Coplanar triangulation edges are excluded.
Crease angles and longest-edge ranking use the source Euclidean geometry before
anisotropic proxy registration.
COMMON ranks all supported cues by one normalized cost; BI prioritizes mask
support for predicted exterior roles and RGB for interior roles. Both use both
cues, the same adoption gates, the same geometry and visibility, and the same
conditional variance rule. Two independent observed incident directions give a
generic 2D mesh junction, one gives a 1D normal factor, and none gives no direct
material observation. Junction factors replace their underlying line evidence;
raw pixel source IDs are consumed once. Original native evidence farther than
eight pixels from the new support is retained. Original downstream H1 role
summaries and feature dimensions are the same for both arms. Overlap tests use
actual original line inliers and supported segments, never a missing display
coordinate. Original normal variance includes its original reliability weight.
Missing source support remains explicitly unknown; a native ID is a semantic
alias and never a certified physical source identity.

Each new arm produces actual poses from the same maximum two physically
distinct N0/E0/PF-priority starts, with at most four updates and ten internal
residual evaluations per update. Analytic projection/mixture Jacobians avoid
hidden numerical-difference residual calls. Every proposed endpoint and rejected
proposal is archived, with availability and guard/fallback reasons. PL consumes
the existing native point/typed-normal/cuboid-outline evidence with this finite
refinement adapter; it is a new objective implementation, not exact old-solver
reuse. CONTOUR uses the actual visible structured proxy silhouette. IMAGE uses
whole native grayscale, rendered diffuse appearance, bidirectional DIS FAST,
non-MSAA metric render depth, back-projected object support, and OpenCV LM pose
refinement. Its rendered background is composited from the observation outside
the rendered object. No GT crop, GT initialization, temporal tracking state, new
network training, or preset tournament is used. Grayscale does not consume all
RGB channel information, and this is not a DeepIM/SRT3D reproduction.

The available EGL device is Mesa llvmpipe, a CPU software renderer. It does not
use CUDA or hardware GPU rendering. An LP_NUM_THREADS=1 timing pilot established
the cost. After measured machine headroom and strict 12-case RGB/depth/mask byte
parity, an explicit resource amendment uses four internal graphics workers in
one renderer process alongside two solver processes; the three dataset workers
and BLAS thread count remain fixed. This changes no model or numeric method.
GL vendor/renderer/version and raster parity against the earlier driver thread
setting are recorded. Pyrender's multisample depth resolve need not correspond
to the centre ray. A
separate non-MSAA visible-depth pass, half-pixel renderer principal-point
conversion, and pose-derived near/far planes give depth on the stored integer
pixel-centre rays. The stored K never changes. An authored slanted-plane test
checks this convention; project/unproject, arbitrary 3D supports, proper proxy
registration, physical-start deduplication, analytic Jacobians, role-independent
dimensionality, and the whole-image depth-correspondence update also have
constructed checks. No current real reference enters these tests or production.

All selected 3D/2D image and contour correspondences are archived. Full raster
arrays are ephemeral; pose/K/asset/source/environment hashes reproduce them.
The campaign method lock binds source closure, environment versions, all assets,
model-identity metadata, and upstream inputs before production. The fixed timing
pilot is first twelve sorted IDs in each domain; completed row checkpoints are
reused exactly in the full 238+1003 run. Technical interruption differs from an
executed correspondence/visibility failure. Physical owner accuracy is NA;
automatic proxy pose generation still executes.

Primary sources checked against the local OpenCV 4.9 API:

- [DIS original paper](https://arxiv.org/abs/1603.03590) and
  [official OpenCV 4.9 DIS API](https://docs.opencv.org/4.9.0/de/d4f/classcv_1_1DISOpticalFlow.html).
  Render-to-real transfer here is an unverified experiment, not a claim from the paper.
- [Official OpenCV 4.9 PnP/refinement API](https://docs.opencv.org/4.9.0/d5/d1f/calib3d_solvePnP.html).
- [Uncertainty-aware points/lines paper, author PDF](https://alexandervakhitov.github.io/scripts/publications/files/vakhitov2021uncertain.pdf).
  The generic covariance-aware adapter is not a reproduction of its full solver.
- [DeepIM](https://arxiv.org/abs/1804.00175), iterative render/image matching motivation only.
- [DLR official 3DObjectTracking](https://github.com/DLR-RM/3DObjectTracking), inspected for a compatible existing tracker.
  No compatible local frozen tracking/refiner path was found; no tracker result is claimed.
