from pathlib import Path
import sys,copy,math
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_role_offset_v35'))
import context35 as V35
from context35 import *
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_continuous_lines_v36'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_continuous_lines_v36.md'
ARMS=('grid','refine','rank','both');STEPS=2000;CANDIDATES=8

def setup36():
    flags=deterministic()
    from sampler34 import install
    install();return flags

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,steps=STEPS,models=12,updates=24000,
      architecture='Frozen V34 seed-specific trunk, parents, query, line encoder/relation; shared diverse8bins/role. Small52->64->64->3 refiner predicts bounded theta/rho residual and quality score. Existing point head is fine-tuned equally in all4arms from same V34 weights. End tokens remain off. Cached visual features fixed; line support center projected coherently onto refined line. Geometry/rank outputs detached on point path: line head supervised only by physical lines, point head by point GT.',
      candidates='Top8 greedy Hough bins per role. Suppress adjacent angle/rho bins with theta period pi and rho sign flip at wrap. All arms same frozen proposals; original V34 top4 model is historical reference, grid arm controls new pool+point-head fine-tuning.',
      refinement='theta residual tanh*pi/24, normalized rho residual tanh*2sqrt2/47; one-bin range. Physical longest visible support >=8px per role inherited V35. Geometry smoothL1 only targets reachable within both residual bounds. Quality target exp(-mean physical support endpoint distance/5px), zero for absent visible role; refined geometry detached when building target.',
      training='Same V32 data/2000 plans:8 legacy+8render eachbatch, train153originals+16renderappearancegroups. AdamW3e-4 wd1e-4 clip5 FP32 strict fixed-grid sampler. Pointmean normalized Euclidean +.05 geometry loss for refine/both +.05 balanced soft quality BCE for rank/both. No check/real GT in fitting. Initialization and batch plan identical withinseed acrossarms.',
      evaluation='Render check same8groups/200requested130detected, same physical support associations as V35. Reused real297 DEV 8conditions; prospective/freshcheck optional diagnostics. Full detection denominators/misses, point error and line geometry quality. Not canonical140/paper/sealed-final evaluation.',
      gates='Both vs grid rendercheck longest-supported point-associated line distance >=10% eachseed and paired8group95%upper<0; same improvement for best-of8 separately shows coordinate potential. Actual point improvement: both vs grid realcrop20outside >=1% eachseed and9sessionCIupper<0, and everyseed clean/crop10/20/30 within1%base, corner20/crop40 retain90%V18gain. All conditions and3seeds regardless result. Geometry success distinct from point stability.',
      probes='At final weights, freeze point/image/candidate visual and swap only line geometry with another session or turn line tokens off; report both realcrop20 changes. No new PCGrad; stop-gradient is fixed experimental isolation.',
      limits='Reuse synthetic check/realDEV andsame2CADassets, rule-defined physical role target. Cached line appearance is not resampled after refinement. Frozen representation/candidate pool reachability may limit learning. No guarantee of improvements or independence from prior pipeline training.'))
    previous=dict(read(V35.RUN/'PRIOR_BINDINGS.json')['files']);previous.update(read(V35.RUN/'COMPLETION.json')['files']);previous[str((V35.RUN/'COMPLETION.json').relative_to(ROOT))]=sha(V35.RUN/'COMPLETION.json');write(RUN/'PRIOR_BINDINGS.json',dict(files=previous))
