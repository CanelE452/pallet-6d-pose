"""Frozen V36 line-to-point information diagnostic; CPU only."""
from pathlib import Path
import os,sys,json,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_line_utility_v37'
V35=ROOT/'data/pallet/results/pallet_role_offset_v35'
V36=ROOT/'data/pallet/results/pallet_continuous_lines_v36'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_line_utility_v37.md'
ARMS=('grid','refine','rank','both');SEEDS=(1,2,3);ANCHORS=('base','v34')
MODES=('point','visual','geometry','visual_geometry')
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
# Import the actual convention rather than assuming edge order.
sys.path.insert(0,str(KIT.parent/'pallet_role_offset_v35'))
from analyze35 import INC,EDGES,ci,fit_offsets

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def population():
    rows=read(V35/'POPULATION.json')['rows'];g=dict(np.load(V35/'geometry.npz'));bb=np.array([r['box'] for r in rows]);wh=np.maximum(bb[:,2:]-bb[:,:2],1)
    return dict(rows=rows,g=g,N=len(rows),bb=bb,diag=np.linalg.norm(wh,axis=-1),center=(bb[:,:2]+bb[:,2:])/2,aspect=np.log(wh[:,0]/wh[:,1]),train=np.array([r['split']=='train' for r in rows]),groups=np.array([r['source_group'] for r in rows]),condition=np.array([r['condition'] for r in rows]),roles=np.load(V35/'analysis_seed1.npz')['selected_role'],supported=np.load(V35/'analysis_seed1.npz')['supported'])

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    p=dict(scope='V37 frozen V36 4arms x 3seeds; two fixed point anchors V12/base and V34. Existing render600requested/436detected, train16/check8 source groups, shared2CAD. No real images/GT evaluation, no neural updates, no previous file modifications.',
    question='Do V36 improved physical line candidates provide additional point correction information, and can a small train-only observable selector recover it?',
    geometry='Canonical unit normal ny>=0 before using V35 train-median physical-to-virtual offset. For each point/incident role/candidate: correction d=n*(rho+median_offset-n.p). Single normal preserves tangent. Report full projections for V35 continuity and bounded .15*boxdiag proposals for operational-selection diagnostics. Oracle targets never enter deployable features.',
    controls='Same frozen base and V34 anchors across all4arms. GT-role longest-support top-score; physical-support GT-best8; matched point-GT-best8+no-op; all24incident candidates+no-op; exact single-normal offset ceiling; multiple physical GT lines+train median/SVD relative .1; multiple GT exact-offset ceiling. Oracles explicitly nondeployable. New line gain evaluated against grid within sameanchor/seed.',
    observable='4ridge gain predictors per arm/anchor/seed =96fits. point: shared frozen32query features averaged over two parents, normalized base/strong/V34/anchor coordinates, aspect, point+role onehot, prior rank, score. visual adds candidate32frozen visual features. geometry adds canonical n, signed line residual, train-median corrected/clipped correction, quadratic/cross terms with anchor-parent differences. visual_geometry adds both. Same bounded24proposals; select max predicted normalized squared-error improvement >0 else no-op. Point/visual names refer to selector input; all proposals still come from lines. Train only16groups, group-balanced weights, ridge lambda.01, feature std floor.05, standardized features clipped[-10,10]. No tuning on check.',
    metrics='Euclidean point error, actual candidate oracle potential and captured gain; normal correction MAE/sign/correlation from line residual; physical-line distance and point-gain correlation; selected harm rates. All detected vs GT-supported, clean/corner/crops/outside/hidden separately. 5000 bootstrap source-group samples with3seeds averaged first, inherited ci routine. Train metrics optimistic because upstream models trained same groups.',
    gates='Primary anchor V34 arm both vs grid: all24+no-op bounded point oracle improves >=1% eachseed and paired8group CI upper<0 => candidate potential added. Observable visual_geometry vs V34 anchor improves >=1% eachseed and CIupper<0 on all_detected, every populated condition/stratum eachseed <=1.01anchor; also beats visual selector with CIupper<0 => information-use screen passed. Gates distinct, no architecture success/real-generalization claim. Otherarm/baseanalyses exploratory; no bestarm selection.',
    decision='If potential not improved, new geometric candidates have not removed point information bottleneck under these proposals. If potential exists but screen fails, correspondence/gain predictability remains; simple ridge failure does not prove nonlinear impossibility. If screen passes, only then separately register constrained learned transfer against samevisual control. No additional longtraining by default.',
    limitations='Reused small check with2sharedCAD; detected-only; physicalGT role visibility in matched/oracle diagnostics only. Selector uses predicted role index not GTvisible role. Normal utility direction only, not full2D vector. Candidate oracle with pointGT can exploit spurious lines. No PCGrad. No manual annotation correction. Fixed bounded proposal family is not universal Hough capacity.')
    write(RUN/'PROTOCOL.json',p)
    prior=read(V36/'PRIOR_BINDINGS.json')['files']|read(V36/'COMPLETION.json')['files'];prior[str((V36/'COMPLETION.json').relative_to(ROOT))]=sha(V36/'COMPLETION.json');write(RUN/'PRIOR_BINDINGS.json',dict(files=prior))
    (RUN/'PURPOSE.md').write_text('# V37 improved-line point residual utility\n\n'+p['question']+'\n\n'+p['scope']+'\n')
    (ROOT/'_docs/handoff/2026-09-10_pallet_line_utility_v37.md').write_text('# V37 선 후보의 점 보정 정보 검증\n\n사용자 승인 후 V36 개선 선 후보를 고정하고 V35 잔차/관측가능성 진단을 반복한다. 합성600관측/436검출, train16/check8그룹, 신경망 업데이트0. 점기준 base/V34 고정,4arm3seed. 정답역할/선oracle/점oracle/GT없는선택 분리. 작은96개 ridge gain식은 train그룹만 학습하고 check로 설정을 바꾸지 않는다. 자세한 사전기준은 results/pallet_line_utility_v37/PROTOCOL.json. 이전결과불변. 완료후MD/시각화/실제Chrome/승인된Discord알림.\n')

if __name__=='__main__':register()
