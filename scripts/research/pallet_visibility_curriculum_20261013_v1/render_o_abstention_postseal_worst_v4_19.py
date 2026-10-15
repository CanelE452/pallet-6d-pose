"""Separate POSTSEAL-worst actual-image diagnostic; never a declared case.

Select the maximum REMAINDER native-only error from completed score rows.
Read stored observations/poses only, with no target pose/corner or new fit.
"""
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import render_o_localization_abstention_process_v4_19 as VIS


C, EXP, OLD, DRAW = VIS.C, VIS.EXP, VIS.OLD, VIS.DRAW
OUTPUT = EXP.OUTPUT.parent/'o_abstention_postseal_worst_v4_19'


def render():
    if OUTPUT.exists():
        raise FileExistsError('Preserve every previous diagnostic')
    root_audit = EXP.OUTPUT/'independent_root_audit_v1'
    audit = C.read(root_audit/'AUDIT.json')
    invariants = C.read(root_audit/'CONTROL_INVARIANTS_V1.json')
    invariant_seal = C.read(root_audit/'CONTROL_INVARIANTS_SEAL_V1.json')
    if (audit['failure_count'] or invariants['failure_count']
            or C.sha(root_audit/'CONTROL_INVARIANTS_V1.json') != invariant_seal['receipt_sha256']
            or audit['prediction_seal']['sha256'] != C.sha(EXP.OUTPUT/'PREDICTION_SEAL.json')):
        raise ValueError('Root independent complete audit required')
    protocol, predictions, scores, reports = OLD._load_completed(EXP.OUTPUT)
    _, parent, parent_scores, _ = OLD._load_completed(EXP.PARENT.OUTPUT)
    fid = max(reports['REMAINDER']['rows'],
        key=lambda r:r['metrics'][VIS.PRIMARY]['translation_cm'])['id']
    if fid in VIS.CASES:
        raise ValueError('This supplement is explicitly separate from predeclared cases')
    task = next(t for t in C.rows(EXP.OUTPUT/'INPUTS.jsonl.gz') if t['id'] == fid)
    observed = next(r for r in C.rows(OLD.OBS.OUTPUT/'OBSERVATIONS.jsonl.gz') if r['id'] == fid)
    old_domains = [r['original_domain'] for r in task['O_localization_abstention']['records']]
    if (observed['states'] != task['states'] or observed['visible_measurements'] != task['spec']['visible']
            or task['new_chain_observation'] != observed['chain_observation']):
        raise ValueError('Use the same source-bound actual observations')
    image_path = C.ROOT/task['image']
    if C.sha(image_path) != task['image_sha256']:
        raise ValueError('Actual original photo hash differs')
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError('Actual original photo decode failed')
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = image.shape[:2]
    keys = (VIS.PARENT_NATIVE, VIS.PRIMARY)
    arms = {keys[0]:parent[fid]['arms'][keys[0]], keys[1]:predictions[fid]['arms'][keys[1]]}
    metrics = {keys[0]:parent_scores[fid]['metrics'][keys[0]], keys[1]:scores[fid]['metrics'][keys[1]]}
    projected = {key:OLD._model(arm, task['spec']['K']) for key, arm in arms.items()}
    geometry = observed['initial_geometry']
    # Viewport derived from unchanged model/BI/O context, never a truth point.
    support = [np.asarray(geometry['initial_projection_xy'])]
    if task['spec']['visible']:
        support.append(np.asarray([r['xy'] for r in task['spec']['visible']]))
    if old_domains:
        support.append(np.asarray([d['center_xy'] for d in old_domains]))
    xy = np.vstack(support); lo = np.maximum(xy.min(0)-35, 0); hi = np.minimum(xy.max(0)+35, [w,h])
    bounds = (float(lo[0]),float(hi[0]),float(hi[1]),float(lo[1]))
    DRAW._font(); OUTPUT.mkdir(parents=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7), squeeze=False)
    ax = axes[0,0]; ax.imshow(image); VIS._context(ax, task); VIS._old_o(ax, old_domains)
    DRAW._wire(ax, np.asarray(geometry['initial_projection_xy']), '#7e8790', ls='--')
    OLD._frozen_points(ax, task)
    ax.set_title('고정 실제 B/I 기록·옛 O 원·U/부분관측\n초기 점선은 모델투영', fontsize=10)
    for index,key in enumerate(keys):
        ax = axes[0,index+1]; ax.imshow(image)
        VIS._pose_panel(ax, projected[key], arms[key], key, metrics[key], bounds, task['states'])
        ax.set_title(VIS.NAMES[key], fontsize=10)
    for ax in axes.flat:
        ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:]); ax.set_aspect('equal'); ax.set_axis_off()
    fig.suptitle('POSTSEAL 추가143 native-only 최대오차 사례: '+fid+
        '\n사전 고정3사례와 별개인 사후 실패 진단 / wire는 저장 모델투영', fontsize=13)
    caption = ('O 원은 raw 예측의 경험 범위이며 실제 가림 물체가 아님. 새19는 O 원 대신 기존 전체 캔버스를 유지. '
        '부분 q는 정밀 RGB KP가 아니며 S 원좌표·U 정밀점·참조/GT wire는 없음. '
        'hard-feasible 수용은 충분한 영상정보·유일한 자세·5cm 정확도 보증이 아님.')
    fig.text(.5,.02,caption,ha='center',fontsize=9,wrap=True)
    fig.tight_layout(rect=(0,.09,1,.9));fig.savefig(OUTPUT/'postseal_worst_actual.png',dpi=160,bbox_inches='tight');plt.close(fig)
    paths = [Path(__file__),Path(VIS.__file__),Path(OLD.__file__),image_path,
        EXP.OUTPUT/'PROTOCOL.json',EXP.OUTPUT/'INPUT_SEAL.json',EXP.OUTPUT/'INPUTS.jsonl.gz',
        EXP.OUTPUT/'PREDICTION_SEAL.json',EXP.OUTPUT/'PREDICTIONS.jsonl.gz',
        EXP.OUTPUT/'SCORE_SEAL.json',EXP.OUTPUT/'REMAINDER_SCORE.json',
        EXP.PARENT.OUTPUT/'PREDICTION_SEAL.json',EXP.PARENT.OUTPUT/'PREDICTIONS.jsonl.gz',
        EXP.PARENT.OUTPUT/'SCORE_SEAL.json',EXP.PARENT.OUTPUT/'REMAINDER_SCORE.json',
        OLD.OBS.OUTPUT/'OBSERVATION_SEAL.json',OLD.OBS.OUTPUT/'OBSERVATIONS.jsonl.gz',
        root_audit/'AUDIT.json',root_audit/'CONTROL_INVARIANTS_V1.json',root_audit/'CONTROL_INVARIANTS_SEAL_V1.json']
    bindings = dict(VIS._binding(p) for p in paths)
    report = dict(schema='O_LOCALIZATION_ABSTENTION_POSTSEAL_WORST_ACTUAL_V4_19',
        id=fid,case_selection='POSTSEAL maximum REMAINDER143 native-only translation; NOT one of the predeclared three cases',
        predeclared_case=False,states=task['states'],visible_count=len(task['spec']['visible']),
        frozen_BI=task['spec']['visible'],partial_boundary=task['boundary_merged'],U_regions=task['u_regions'],
        original_O_domains=old_domains,replacement_O_T_domains=task['spec']['domains'],
        initial_geometry=geometry,image=task['image'],image_sha256=task['image_sha256'],crop_bounds=bounds,
        methods={key:dict(actual_pose=arms[key]['actual_pose'],accepted=arms[key]['accepted'],
            WD_family=arms[key].get('WD_family'),metric=metrics[key],
            model_projection_xy=None if projected[key] is None else projected[key].tolist(),
            original_O_projection_audit=VIS._domain_audit(projected[key],old_domains,[h,w])) for key in keys},
        image_support_does_not_certify_unique_pose=True,no_target_pose_or_corner_read_or_drawn=True,
        S_raw_not_read=True,U_not_precise_point=True,new_inference_or_fit=0,
        default_e_unchanged=True,source_bindings=bindings)
    C.write(OUTPUT/'POSTSEAL_WORST_DIAGNOSTIC.json',report)
    for path,digest in bindings.items():
        if C.sha(C.ROOT/path)!=digest:
            raise ValueError('Bound source changed during diagnostic')
    C.write(OUTPUT/'RENDER_SEAL.json',dict(source_sha256=C.sha(Path(__file__)),input_sha256=bindings,
        assets={str(p.relative_to(OUTPUT)):C.sha(p) for p in OUTPUT.rglob('*') if p.is_file()},
        postseal_case_selection=True,predeclared_case=False,new_inference_or_fit=0,no_GT_wire=True))
    print(dict(output=str(OUTPUT),postseal_maximum_id=fid),flush=True)


if __name__=='__main__':
    render()
