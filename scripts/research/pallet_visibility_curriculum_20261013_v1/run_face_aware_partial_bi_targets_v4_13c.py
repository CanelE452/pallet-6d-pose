"""Two preregistered actual-image partial-BI predictions; no pose/GT scoring."""
from __future__ import annotations
import os
for _thread_key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_thread_key] = '1'
from copy import deepcopy
from pathlib import Path
import shutil
import time
import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import run_global_BI_edges_v4_9c as D
from . import face_aware_partial_bi_observations_v4_13c as P
from . import statewise_partial_BI_u_solver_v4_13 as SOLVER

TARGETS = ('eval_noapril:1775201430551447040', 'eval_pallet07:1778652140531310080')
OUTPUT = D.OUTPUT.parent/'face_aware_partial_bi_targets_v4_13c'
MODEL = OUTPUT.parent.parent/'truncation_interval_model_CAL_v4_2/MODEL.json'


def _figure(row, observation, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = sorted(Path('/usr/share/fonts').rglob('*NanumGothic*.ttf'))
    if fonts:
        font_manager.fontManager.addfont(str(fonts[0]))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(fonts[0])).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    image = cv2.imread(str(C.ROOT/row['image']))[..., ::-1]
    old = {m['native_corner']:m for m in row['frozen_BI_fit']}
    selected = observation['boundary_observations']
    ids = ([0,1,3] if row['id'].startswith('eval_noapril:') else
           sorted({b['native_corner'] for b in selected})[:3])
    if not ids:
        ids = [m['native_corner'] for m in row['frozen_BI_fit']][:3]
    fig, axes = plt.subplots(1, 1+len(ids), figsize=(5+3.7*len(ids), 4.3), squeeze=False)
    def draw(ax, focus=None):
        ax.imshow(image)
        for k in ids:
            q = np.asarray(old[k]['xy'])
            if focus is None or k == focus:
                ax.plot(*q, marker='+', color='#d325d3', ms=10, mew=1.7)
                ax.annotate(f'{row["states"][k]}{k} 기존 보관값', q, xytext=(6,7), textcoords='offset points', color='#a010a0', fontsize=9)
        for b in selected:
            k = b['native_corner']
            if focus is not None and k != focus:
                continue
            for line in b['lines']:
                ray = line['provenance']['line_record']
                points = np.asarray(ray['observed_points']); points = points[np.asarray(ray['inlier_mask'],bool)]
                n = np.asarray(line['normal_unit_xy']); tangent = np.array([-n[1],n[0]])
                lo,hi = ray['actual_inlier_tangent_interval_px']
                segment = -line['offset_px']*n + np.asarray([lo,hi])[:,None]*tangent
                ax.plot(segment[:,0],segment[:,1],color='#008f4a',lw=2)
                ax.scatter(points[:,0],points[:,1],c='#008f4a',s=10)
            display = np.asarray(b['association_provenance']['normal_projected_display_xy'])
            ax.plot(*display,marker='s',color='#e07600',mfc='none',ms=7)
            ax.annotate(f'{row["states"][k]}{k} 법선 투영 표시',display,xytext=(5,-15),textcoords='offset points',color='#c15b00',fontsize=9)
        if focus is not None:
            q = np.asarray(old[focus]['xy']); ax.set_xlim(q[0]-65,q[0]+65);ax.set_ylim(q[1]+55,q[1]-55)
        ax.set_aspect('equal');ax.set_axis_off()
    draw(axes[0,0]);axes[0,0].set_title('실제 RGB · 초록선은 실제 인라이어 구간',fontsize=11)
    for ax,k in zip(axes[0,1:],ids):
        draw(ax,k);ax.set_title(f'{row["states"][k]}{k}: 부분 법선 / 원본 q 보관',fontsize=11)
    fig.suptitle(row['id']+'\n주황 사각형은 완전한 2D KP가 아님 · S/U 코너 영상 조회 없음 · GT/새 PnP 사용 없음',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.89));path=output/(row['id'].replace(':','__')+'.png')
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    return path


def run(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing target predictions '+str(output))
    rows, _, _, bindings, parent, old = D._load()
    by_id = {r['id']:r for r in rows}
    population = [by_id[k] for k in TARGETS]
    localization = C.read(MODEL)
    sources = deepcopy(old['source_sha256'])
    for module in (P,SOLVER):
        path = Path(module.__file__);sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    for path in (Path(__file__),Path(P.__file__).with_name('test_face_aware_partial_bi_observations_v4_13c.py')):
        sources[str(path.relative_to(C.ROOT))] = C.sha(path)
    bindings[str(MODEL.relative_to(C.ROOT))]=C.sha(MODEL)
    protected = parent['protected_user_sha256']
    for v in (sources, bindings, protected):
        E._verify_sources(v)
    output.mkdir(parents=True)
    protocol = dict(schema='FACE_AWARE_PARTIAL_BI_ACTUAL_TWO_TARGETS_V4_13C',ids=list(TARGETS),
        source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,
        configuration=P.DEFAULT_CONFIGURATION,face_tolerance_deg=P.FACE_TOLERANCE_DEG,
        fixed_old_V47_q_observedSigma_effectiveSigma_reliability=True,
        mask='sealed V46 full clean MobileSAM mask; no new forward',
        states_model_roles_initialS_K_Q_OT_U_unchanged=True,
        unknown_neighbors_are_model_direction_only_no_raw_or_corner_feature=True,
        declared_before_current_partial_fit=True,GT_files_opened=False,pose_solves=0,
        target_selection='previously diagnosed failure mechanisms; TEST reused exploratory, not generalization evidence')
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        dest=output/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,dest)
        if C.sha(dest)!=digest:
            raise ValueError('Source snapshot changed')
    values=[];started=time.monotonic()
    for row in population:
        image=cv2.imread(str(C.ROOT/row['image']))
        with np.load(row['_mask_path'],allow_pickle=False) as z:
            mask=z['pallet_mask_clean'].copy()
        raw=row['_original_source']['original_raw_points']
        scale=row['partial_boundary_observation']['unchanged_nonself_prediction_scale_px']
        fit=row['frozen_BI_fit'];before=E._digest(fit)
        observation=P.observe_face_aware_partial_bi(image,mask,row['runtime']['initial_geometry'],row['states'],fit,
            raw_native_points=raw,localization_model=localization,nonself_scale_px=scale,
            image_binding=dict(sha256=row['image_sha256'],image_hw=list(image.shape[:2]),relative_path=row['image']))
        if E._digest(fit)!=before or observation['visible_measurements']!=fit:
            raise ValueError('Original full-BI observations changed')
        normalized=SOLVER._boundary_inputs(observation['boundary_observations'],row['states'],image.shape[:2],
            np.asarray(row['runtime']['initial_geometry']['native_xyz']))
        ids=SOLVER.validate_complete_BI_likelihood_revocations(normalized,fit)
        value=C.finite(dict(id=row['id'],partition=row['partition'],image=row['image'],image_sha256=row['image_sha256'],
            initial_geometry=row['runtime']['initial_geometry'],states=row['states'],frozen_BI_fit=fit,
            partial_observation=observation,normalized_revoked_BI_ids=ids,GT_inputs=False,pose_solves=0))
        C.write(output/(row['id'].replace(':','__')+'.json'),value);values.append(value)
        print('ACTUAL_PARTIAL_TARGET',row['id'],'partial',ids,flush=True)
    for v in (sources,bindings,protected):
        E._verify_sources(v)
    C.write(output/'PREDICTIONS.json',values)
    C.write(output/'PREDICTION_SEAL.json',dict(rows=2,predictions_sha256=C.sha(output/'PREDICTIONS.json'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),source_sha256=sources,
        GT_files_opened=False,pose_solves=0,elapsed_seconds=time.monotonic()-started))
    figures=[_figure(row,value['partial_observation'],output) for row,value in zip(population,values)]
    C.write(output/'RECEIPT.json',dict(prediction_seal_sha256=C.sha(output/'PREDICTION_SEAL.json'),
        figure_sha256={p.name:C.sha(p) for p in figures},typed_archive_validator_passed=True,
        source_model_input_protected_hashes_unchanged=True,GT_files_opened=False,new_pose_fits=0,
        new_mask_or_state_forward=0,full_corner_measurements_created=0))


if __name__=='__main__':
    run()
