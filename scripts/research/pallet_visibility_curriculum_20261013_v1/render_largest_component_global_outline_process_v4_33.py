"""Postscore fixed V32 cases: union/LCC pixels, stored evidence and final V33.

No new reference document is opened. The last separate reference wire is the
already archived V32 PROCESS_DATA projection. Viewports use only the frozen
observed/coarse/support ROI rule, never reference/final-pose/error geometry.
No NN, native start search, TLS, pose fit or intermediate trajectory is run.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import shutil
import sys

import numpy as np

from . import score_largest_component_global_outline_poses_v4_33 as SCORE
from . import render_clean_fresh_BI_full_result_process_v4_32_r2_v1 as OLD
from . import render_saved_full_result_roi_v4_32_r2_v1 as ROI

C = SCORE.C
M = SCORE.M
LCC = M.LCC
DRAW = OLD.DRAW
OUTPUT = M.B/'largest_component_global_outline_process_v4_33_v1'
PLAN = M.B/'largest_component_global_outline_process_plan_v4_33/PREDECLARATION.json'
PARENT = OLD.OUTPUT
PARENT_RENDER_SEAL_SHA256 = '78bc8404d5bf4c1a6de85c60b549d1a2e32ab16979f80671463506cc1cb28a9a'
OLD_RENDERER_SHA256 = 'a7006dbbcaf08f91528f72b2bea21787be3d74fcf8d9406d551c46547d4bbf39'
SCORER_SHA256 = 'd9871f83e9e7207d4ba02edddc18d7b531b03bc84e818cdaf3ea41c25e901748'
FIXED_IDS = ('wood_night_01:031344', 'wood_night_01:031426', 'eval_noapril:1775201428535961600')
ARMS = M.ARMS
observed_model_ROI = ROI.observed_model_ROI
final_projection = OLD.final_projection


def _sources():
    paths = [Path(__file__),Path(__file__).with_name('test_render_largest_component_global_outline_process_v4_33.py'),
        Path(SCORE.__file__),Path(M.__file__),Path(OLD.__file__),Path(ROI.__file__),Path(DRAW.__file__),Path(LCC.__file__)]
    return M._union(SCORE._sources(),{M._relative(path):C.sha(path) for path in paths})


def _pins():
    if C.sha(OLD.__file__) != OLD_RENDERER_SHA256 or C.sha(SCORE.__file__) != SCORER_SHA256:
        raise ValueError('Frozen archived process renderer and reviewed scorer required')
    SCORE._pins()


def _peer(binding):
    from . import declare_initial_only_observed_ray_v4_32 as D
    path = D._peer(binding,(M,SCORE,OLD,sys.modules[__name__]))
    test = Path(__file__).with_name('test_render_largest_component_global_outline_process_v4_33.py')
    if D._source_bindings(C.read(path)).get(M._relative(test)) != C.sha(test):
        raise ValueError('Renderer peer must bind the exact new toy source')
    return path


def _completed(central_sha, prediction_sha, score_sha):
    _pins()
    protocol = SCORE._prediction_gate(M.OUTPUT,central_sha,prediction_sha)
    if C.sha(M.OUTPUT/'SCORE_SEAL.json') != score_sha:
        raise ValueError('Caller-pinned completed V33 score seal required')
    seal = C.read(M.OUTPUT/'SCORE_SEAL.json')
    scoring = C.read(M.OUTPUT/'SCORING_PROTOCOL.json')
    if (seal['prediction_seal'] != M._binding(M.OUTPUT/'PREDICTION_SEAL.json')
            or seal['scoring_protocol_sha256'] != C.sha(M.OUTPUT/'SCORING_PROTOCOL.json')
            or seal['outline_selection_profile'] != LCC.SELECTION_PROFILE
            or seal['native_CAL32_models_relabelled'] is not False
            or seal['all_rows_kept'] is not True or seal['default'] != 'SAME_FROZEN_E unchanged'
            or set(seal['score_sha256']) != {*M.PARTITIONS,'PRIMARY'}
            or any(C.sha(M.OUTPUT/(part+'_SCORE.json')) != digest for part,digest in seal['score_sha256'].items())
            or scoring['central_plan_sha256'] != central_sha or scoring['prediction_seal_sha256'] != prediction_sha
            or scoring['outline_selection_profile'] != LCC.SELECTION_PROFILE
            or scoring['native_CAL32_models_relabelled'] is not False):
        raise ValueError('Exact same central/current prediction/fixed-reference V33 scores required')
    M._verify(*(scoring[key] for key in ('source_sha256','input_sha256','protected_user_sha256')))
    if C.sha(PARENT/'RENDER_SEAL.json') != PARENT_RENDER_SEAL_SHA256:
        raise ValueError('Completed frozen V32 three-case process required')
    parent = C.read(PARENT/'RENDER_SEAL.json')
    cases = {row['id']:row['folder'] for row in parent['cases']}
    if len(cases) != 3 or set(cases) != set(FIXED_IDS):
        raise ValueError('Same three previously selected cases, no new error routing')
    inputs = {M._relative(PARENT/'RENDER_SEAL.json'):PARENT_RENDER_SEAL_SHA256}
    for identifier in FIXED_IDS:
        for name in ('PROCESS_DATA.json','source_rgb.png'):
            relative=cases[identifier]+'/'+name;path=PARENT/relative
            if C.sha(path) != parent['assets'][relative]:
                raise ValueError('Archived process or original copied RGB changed')
            inputs[M._relative(path)] = C.sha(path)
    return protocol,scoring,cases,inputs


def declare(*, root_metadata_GO=False, expected_central_plan_sha256=None,
            expected_prediction_seal_sha256=None, expected_score_seal_sha256=None,
            source_peer_binding=None):
    if root_metadata_GO is not True:
        raise PermissionError('ROOT renderer metadata GO before IO')
    if (not all(M._sha(value) for value in (expected_central_plan_sha256,expected_prediction_seal_sha256,expected_score_seal_sha256))
            or not isinstance(source_peer_binding,dict) or set(source_peer_binding) != {'path','sha256'}
            or not isinstance(source_peer_binding['path'],str) or not M._sha(source_peer_binding['sha256'])):
        raise ValueError('Explicit central/PRED/score/source-peer bindings before IO')
    if PLAN.exists():
        raise FileExistsError('Preserve each V33 render declaration')
    peer = _peer(source_peer_binding)
    protocol,scoring,cases,inputs = _completed(expected_central_plan_sha256,
        expected_prediction_seal_sha256,expected_score_seal_sha256)
    inputs = M._union(inputs,{M._relative(peer):C.sha(peer)},
        {M._relative(M.OUTPUT/name):C.sha(M.OUTPUT/name) for name in ('INPUTS.jsonl.gz','PREDICTIONS.jsonl.gz',
            'PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json','SCORE_SEAL.json','SCORING_PROTOCOL.json','PRIMARY_SCORE.json')},
        {M._relative(M.OUTLINES/name):C.sha(M.OUTLINES/name) for name in ('OUTLINE_OBSERVATION_SEAL.json','OUTLINE_OBSERVATIONS.jsonl.gz')})
    sources=_sources()
    value=dict(schema='FIXED_THREE_CASE_UNION_LCC_PROCESS_PLAN_V33',UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protocol['protected_user_sha256'],
        central_plan_sha256=expected_central_plan_sha256,prediction_seal_sha256=expected_prediction_seal_sha256,
        score_seal_sha256=expected_score_seal_sha256,source_peer_binding=source_peer_binding,
        fixed_ids=list(FIXED_IDS),parent_case_folders=cases,outline_selection_profile=LCC.SELECTION_PROFILE,
        case_selection='same previously selected V32 postscore three cases; no new V33 worst-case or method routing',
        panels=['full original RGB / union target-mask pixels and stored outline / LCC pixels and stored outline',
            'same observed/model ROI / archived coarse BI and U/O ranges / union and LCC observed boundary / actual finite line constraints',
            'five stored V33 predictions; C panels compare archived union prediction as gray dots / last separate already archived fixed-reference projection'],
        ROI='exact old observed_model_ROI; original BI+coarse+actual support only; no GT/final pose/score geometry',
        reference='old sealed PROCESS_DATA.fixed_reference_projection_xy only; no reference document reopened',
        mask_pixels='exact frozen LCC binary component selection replay for display, checked against saved pixel digests; not a new observation',
        native_population_and_CAL32_noise_unchanged=True,S_observed_points=0,U_O_point_markers=0,
        no_new_NN_TLS_PnP_optimizer_or_precise_keypoints=True,unarchived_start_and_continuous_trajectory_not_invented=True,
        semantic_target_or_occluder_identity_certified=False,no_accuracy_or_adoption_claim=True,default='SAME_FROZEN_E unchanged')
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN,value)
    C.write(PLAN.with_name('PLAN_SEAL.json'),dict(plan_sha256=C.sha(PLAN),source_sha256=sources))
    return M._binding(PLAN)


def verify_display_mask(mask, outline):
    """Display-only exact pixel replay; no RGB query or sampler/TLS calls."""
    selected,audit=LCC.largest_component_mask(mask,outline['largest_component']['image_hw'])
    if audit != outline['largest_component']['foreground_component_selection']:
        raise ValueError('Display LCC pixel replay differs from sealed observation')
    original_float=np.ascontiguousarray(np.asarray(mask,dtype=float))
    selected_float=np.ascontiguousarray(np.asarray(selected,dtype=float))
    for pixels,observed in ((original_float,outline['original_union']),(selected_float,outline['largest_component'])):
        if hashlib.sha256(pixels.tobytes()).hexdigest() != observed['mask_array_sha256']:
            raise ValueError('Exact original/selected mask raster binding required')
    return (np.asarray(mask)>=.5),selected.astype(bool),audit


def verify_archived_context(packet, data):
    """Compare already admitted original observations; never read S/U q."""
    source=packet['clean_input']
    actual=dict(states=source['states'],original_native_xyz=source['native_xyz'],original_Q=source['physical_registration_Q'],
        K=source['K'],own_independent_coarse=source['initial_coarse_pose'],original_BI_archives=OLD.archived_BI(packet),
        U_ranges=packet['physics']['u_regions'],O_regions=packet['regularizers']['o_regions'])
    if any(M._digest(value) != M._digest(data[key]) for key,value in actual.items()):
        raise ValueError('Original archived coarse/BI q-source-cov/state/ranges/physical geometry must remain exact')
    return actual


def _outline(ax, observation, color, label):
    if not observation['supported']:
        ax.text(.03,.07,'UNSUPPORTED — retained',transform=ax.transAxes,color=color,fontsize=10)
        return
    for index,segment in enumerate(observation['observed_boundary_segments_xy']):
        xy=np.asarray(segment)
        ax.plot(xy[:,0],xy[:,1],color=color,linewidth=1.5,label=label if index==0 else None)


def _fixed_packets(path):
    seen=set()
    for packet in C.rows(path):
        identifier=packet['identifiers']['id']
        if identifier not in FIXED_IDS:
            del packet
            continue
        if identifier in seen:
            raise ValueError('Duplicate fixed case positive packet')
        seen.add(identifier)
        yield packet
    if seen != set(FIXED_IDS):
        raise ValueError('All three fixed explanation cases must remain')


def render(*, root_actual_postscore_render_GO=False):
    if root_actual_postscore_render_GO is not True:
        raise PermissionError('ROOT full postscore actual render GO before numeric/image IO')
    if OUTPUT.exists():
        raise FileExistsError('Preserve every V33 actual process render')
    plan=C.read(PLAN)
    if C.read(PLAN.with_name('PLAN_SEAL.json'))['plan_sha256'] != C.sha(PLAN):
        raise ValueError('Exact frozen V33 process plan required')
    M._verify(plan['source_sha256'],plan['input_sha256'],plan['protected_user_sha256'])
    protocol,scoring,cases,inputs=_completed(plan['central_plan_sha256'],plan['prediction_seal_sha256'],plan['score_seal_sha256'])
    scores=C.read(M.OUTPUT/'PRIMARY_SCORE.json')
    if scores['count'] != 192 or len(scores['rows']) != 192 or scores['all_rows_kept'] is not True:
        raise ValueError('Saved full PRIMARY192 scores required, no new scoring')
    metrics={row['id']:row['metrics'] for row in scores['rows'] if row['id'] in FIXED_IDS}
    OUTPUT.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    cards={};saved=[]
    try:
        for packet in _fixed_packets(M.OUTPUT/'INPUTS.jsonl.gz'):
            identifier=packet['identifiers']['id'];ci=FIXED_IDS.index(identifier);folder=OUTPUT/('case'+str(ci));folder.mkdir()
            oldfolder=PARENT/cases[identifier];data=C.read(oldfolder/'PROCESS_DATA.json')
            verify_archived_context(packet,data)
            outline=OLD._only_row(M.OUTLINES/'OUTLINE_OBSERVATIONS.jsonl.gz',identifier)
            row=OLD._only_row(M.OUTPUT/'PREDICTIONS.jsonl.gz',identifier)
            if outline['selection_profile'] != LCC.SELECTION_PROFILE or outline['union_exact_match'] is not True:
                raise ValueError('Sealed same-case union/LCC observation required')
            union={key:value for key,value in outline['original_union'].items() if key!='selection_profile'}
            if M._digest(packet['regularizers']['outline_observation']) != M._digest(outline['largest_component']):
                raise ValueError('Current C observation differs from saved LCC row')
            binding=outline['target_mask_binding'];maskpath=C.ROOT/binding['path']
            if C.sha(maskpath) != binding['sha256']:
                raise ValueError('Same original cached mask file required')
            with np.load(maskpath,allow_pickle=False) as arrays:
                mask=arrays['pallet_mask_clean'].copy()
            union_mask,lcc_mask,audit=verify_display_mask(mask,outline)
            image=np.asarray(Image.open(oldfolder/'source_rgb.png').convert('RGB'))
            if tuple(image.shape[:2]) != tuple(packet['physics']['image_hw']) or C.sha(oldfolder/'source_rgb.png') != packet['identifiers']['image_sha256']:
                raise ValueError('Exact original RGB/canvas required')
            shutil.copyfile(oldfolder/'source_rgb.png',folder/'source_rgb.png')
            inputs[M._relative(maskpath)]=binding['sha256']
            roi=observed_model_ROI(data,image.shape[:2]);x0,x1,y0,y1=roi
            def photo(ax,title,crop=True):
                DRAW._photo(ax,image,title)
                if crop:ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
            fig,axes=plt.subplots(1,3,figsize=(19,6),layout='constrained')
            photo(axes[0],'Actual original RGB — full canvas',False)
            for ax,pixels,observed,color,title in ((axes[1],union_mask,union,'#ffbd45','Original union target-mask pixels + stored outline'),
                (axes[2],lcc_mask,outline['largest_component'],'#00ffff','LCC8 pixels + stored outline')):
                photo(ax,title,False)
                rgba=np.zeros((*pixels.shape,4));rgba[pixels]=(*matplotlib.colors.to_rgb(color),.27)
                ax.imshow(rgba);_outline(ax,observed,color,'stored convex foreground observation')
                if observed['supported']:ax.legend(fontsize=8)
            fig.suptitle(identifier+f"\nComponents={audit['foreground_component_count']}; retained={audit['retained_foreground_area_px']}px; discarded={audit['discarded_foreground_area_px']}px — no semantic ownership proof")
            fig.savefig(folder/'union_LCC_full_canvas.png',dpi=140);plt.close(fig)
            fig,axes=plt.subplots(3,2,figsize=(16,15));fig.subplots_adjust(top=.92,bottom=.06,hspace=.58,wspace=.22);axes=axes.ravel()
            photo(axes[0],'Actual RGB — same observed/model ROI')
            photo(axes[1],'Own coarse PREDICTION + archived B/I + U/O ranges')
            DRAW._wire(axes[1],np.asarray(data['coarse_projection_xy']),color=DRAW.PURPLE,label='predicted own coarse')
            DRAW._BI(axes[1],data['original_BI_archives']);OLD._ranges(axes[1],packet)
            photo(axes[2],'Original union — stored mask boundary, not native edge')
            _outline(axes[2],union,'#ffbd45','union stored mask observation')
            photo(axes[3],'LCC8 — stored mask boundary, not native edge')
            _outline(axes[3],outline['largest_component'],'#00ffff','LCC stored mask observation')
            for ax,arm in zip(axes[4:],M.NATIVE_ARMS):
                entry=row['arms'][arm];photo(ax,arm.replace('CLEAN_FRESH_BI_','')+' — unchanged native evidence')
                active=[];owners=[]
                if entry['pose_available']:
                    activeplan=entry['new_geometry_edge_cost_plans'][entry['WD_family']]
                    active=activeplan['active_groups'];owners=activeplan['new_owner_ids'];DRAW._support(ax,active)
                ax.text(.01,-.19,f'active={len(active)}; owners={owners}\nActual finite support; C retains this policy population, may select another WD.',transform=ax.transAxes,va='top',fontsize=10)
            fig.suptitle(identifier+' — stored observations and unchanged native support; no new KP or intermediate path')
            fig.savefig(folder/'stored_context_and_boundaries_ROI.png',dpi=145);plt.close(fig)
            fig,axes=plt.subplots(3,2,figsize=(16,15));fig.subplots_adjust(top=.92,bottom=.06,hspace=.58,wspace=.22);axes=axes.ravel()
            projections={};finals={}
            for ax,arm,color in zip(axes[:5],ARMS,OLD.COLORS):
                entry=row['arms'][arm];photo(ax,arm.replace('CLEAN_FRESH_BI_','')+' — PREDICTED')
                xy=final_projection(packet,entry,arm,row['legacy_e_pose_semantics']);projections[arm]=None if xy is None else xy.tolist()
                if arm in tuple(names[1] for names in M.M.POLICY_ARMS.values()):
                    previous=data['final_projection_xy'][arm]
                    if previous is not None:DRAW._wire(ax,np.asarray(previous),color='#c4c4c4',label='stored V32 union prediction',style=':',width=1.4)
                if xy is not None:DRAW._wire(ax,xy,color=color,label='stored final physical-body prediction',style='-',width=1.8)
                pose=entry.get('actual_pose');wd=entry.get('WD_family',None if pose is None else pose.get('WD_family'))
                v=metrics[identifier][arm]
                finals[arm]=dict(actual_pose=deepcopy(pose),accepted=entry['accepted'],pose_available=entry.get('pose_available',True),WD_family=wd)
                active_count=(None if arm==ARMS[0] or not entry.get('pose_available',True) else
                    len(entry['new_geometry_edge_cost_plans'][entry['WD_family']]['active_groups']))
                ax.text(.01,-.19,f"T={v['translation_cm']} cm; R={v['rotation_deg']} deg\naccepted={entry['accepted']}; WD={wd}; new active={active_count}",transform=ax.transAxes,va='top',fontsize=10)
                if arm in tuple(names[1] for names in M.M.POLICY_ARMS.values()):
                    ax.legend(loc='upper right',fontsize=8)
            photo(axes[-1],'ARCHIVED FIXED EVALUATION REFERENCE ONLY')
            reference=data['fixed_reference_projection_xy']
            if reference is not None:DRAW._wire(axes[-1],np.asarray(reference),color='#5bff77',label='archived evaluation reference; never a guide',style='-',width=1.8)
            else:axes[-1].text(.03,.1,'Archived reference R unresolved; wire omitted',transform=axes[-1].transAxes)
            axes[-1].text(.01,-.19,'Last separate panel; no reference document reopened.\nDerived references are not independent physical metrology.',transform=axes[-1].transAxes,va='top',fontsize=10)
            fig.suptitle(identifier+' — stored final V33 predictions; native WD channels are not equated')
            fig.savefig(folder/'final_predictions_and_archived_reference_ROI.png',dpi=145);plt.close(fig)
            proof=dict(id=identifier,partition=row['partition'],outline_selection_profile=LCC.SELECTION_PROFILE,
                parent_process_binding=M._binding(oldfolder/'PROCESS_DATA.json'),image_binding=M._binding(oldfolder/'source_rgb.png'),
                target_mask_binding=binding,display_component_replay=audit,ROI_original_xy=roi,
                original_union_observation=union,LCC_observation=outline['largest_component'],
                native_active_counts={arm:0 if not row['arms'][arm]['pose_available'] else len(row['arms'][arm]['new_geometry_edge_cost_plans'][row['arms'][arm]['WD_family']]['active_groups']) for arm in ARMS[1:]},
                archived_context=verify_archived_context(packet,data),coarse_projection_xy=data['coarse_projection_xy'],
                final_predictions=finals,final_projection_xy=projections,saved_metrics=metrics[identifier],
                archived_union_C_projection_xy={names[1]:data['final_projection_xy'][names[1]] for names in M.M.POLICY_ARMS.values()},
                archived_union_C_saved_metrics={names[1]:data['saved_metrics'][names[1]] for names in M.M.POLICY_ARMS.values()},
                archived_reference_projection_xy=reference,archived_reference_binding=data['fixed_reference_binding'],
                stages=dict(original_RGB=True,archived_BI_and_ranges=True,independent_coarse=True,stored_native_lines=True,final_poses=True,
                    native_start_R_t=False,continuous_intermediate_path=False),S_observed_points=0,U_O_center_markers=0,
                no_new_reference_NN_TLS_PnP_optimizer_or_image_feature_queries=True,semantic_component_identity_proved=False)
            C.write(folder/'PROCESS_DATA.json',proof);saved.append(dict(id=identifier,folder=folder.name))
            cards[identifier]=['## '+identifier,'',f'![원본과 union/LCC 전체 픽셀]({folder.name}/union_LCC_full_canvas.png)','',
                f'![동일 ROI의 보관 관측]({folder.name}/stored_context_and_boundaries_ROI.png)','',
                f'![최종 예측과 별도 보관 참조]({folder.name}/final_predictions_and_archived_reference_ROI.png)','',
                f'[보관 값·출처]({folder.name}/PROCESS_DATA.json) · [원본 RGB]({folder.name}/source_rgb.png)','']
            del packet,row,outline,data,image,mask,union_mask,lcc_mask
        lines=['# 고정 세 사례: union과 LCC8 전경 포함 비용 비교','',
            '기존 V32 설명에 쓴 세 사례를 그대로 고정했다. V33 오차를 보고 다시 고르지 않았으며 이 그림은 전수 성공률·방법 선택의 근거가 아니다. 기본값은 SAME_FROZEN_E다.','',
            '원래 union과 LCC8은 같은 캐시 마스크에서 나온다. LCC8은 가장 큰 8연결 성분만 남긴 고정 규칙이다. 큰 성분이 파렛트라는 증명은 없으며, 가림으로 분리된 진짜 파렛트 조각을 버릴 수도 있다. 다른 물체·가림의 정답 마스크로 해석하지 않는다.','',
            '보라 모델선은 예측이고 B/I 점은 N3/SubPix/BI 보정 archive다. S는 모델만, U/O는 대략 범위만 표시하며 중심을 관측점으로 그리지 않았다. 실제 native 유한 선 지지와 전경 마스크 경계는 구분했다. native 선·noise32는 그대로이며 active=0은 빈 채로 표시했다.','',
            '최종 패널의 실선은 저장된 V33 예측이다. C 패널의 회색 점선은 기존 V32 union 예측이며 보정 전후의 중간 자세가 아니다. native 두 방법과 e는 원 결과와 정확히 같아야 하는 대조다. 마지막 초록 패널만 기존 봉인 PROCESS_DATA에 이미 있던 평가 참조다. 새 참조 문서를 열거나 참조로 ROI·탐색·보정·fit을 하지 않았다. 보관되지 않은 native 시작 자세와 연속 최적화 중간 궤적은 새로 계산하거나 만들지 않았다.','']
        for identifier in FIXED_IDS:lines+=cards[identifier]
        (OUTPUT/'REPORT_KO.md').write_text('\n'.join(lines))
        M._verify(plan['source_sha256'],plan['input_sha256'],plan['protected_user_sha256'])
        C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='SEALED_FIXED_THREE_CASE_UNION_LCC_PROCESS_V33',plan_sha256=C.sha(PLAN),
            source_sha256=plan['source_sha256'],input_sha256=M._union(plan['input_sha256'],inputs),cases=saved,
            assets={str(path.relative_to(OUTPUT)):C.sha(path) for path in sorted(OUTPUT.rglob('*')) if path.is_file()},
            prediction_seal_sha256=plan['prediction_seal_sha256'],score_seal_sha256=plan['score_seal_sha256'],
            fixed_reference_document_decodes=0,new_NN_TLS_PnP_optimizer_or_feature_queries=0,S_observed_points=0,U_O_center_markers=0,
            mask_component_replay_for_display_only=True,semantic_target_or_occlusion_identity_certified=False,
            fixed_case_explanation_only=True,default='SAME_FROZEN_E unchanged',adoption=False,visual_QA_pending=True))
    except BaseException as error:
        C.write(OUTPUT/'FAILURE_RECEIPT_V1.json',dict(exception=type(error).__name__,reason=str(error),
            completed_cases=saved,attempt_preserved=True,silently_modified_or_restarted=False))
        raise
    return M._binding(OUTPUT/'RENDER_SEAL.json')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('declare','render'))
    parser.add_argument('--root-metadata-GO',action='store_true')
    parser.add_argument('--root-actual-postscore-render-GO',action='store_true')
    parser.add_argument('--central-plan-sha256');parser.add_argument('--prediction-seal-sha256');parser.add_argument('--score-seal-sha256')
    parser.add_argument('--source-peer');parser.add_argument('--source-peer-sha256')
    args=parser.parse_args()
    if args.stage=='declare':
        print(declare(root_metadata_GO=args.root_metadata_GO,expected_central_plan_sha256=args.central_plan_sha256,
            expected_prediction_seal_sha256=args.prediction_seal_sha256,expected_score_seal_sha256=args.score_seal_sha256,
            source_peer_binding=dict(path=args.source_peer,sha256=args.source_peer_sha256)))
    else:print(render(root_actual_postscore_render_GO=args.root_actual_postscore_render_GO))
