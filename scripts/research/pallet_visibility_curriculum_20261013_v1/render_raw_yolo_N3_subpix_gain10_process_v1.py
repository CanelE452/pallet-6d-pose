"""Saved TEST49 paired gain gallery: actual cached 2D stages and F poses only.

No F/SQPnP/LM, neural network, subpixel call, visibility classifier, mask or
reference loader is called. Intermediate qN/qS/qFinal are points, not poses.
All detector points below are predictions, including occluded corner channels.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import html
from pathlib import Path
import shutil

import numpy as np

from . import common as C
from . import remainder_native_inputs_v1 as SELECT

B = C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OUTPUT = B/'raw_yolo_N3_subpix_gain10_process_v1'
COMPARISON = C.DOC/'nonsevere_v1/posefix_sub_comparison_v1'
COHORT = C.DOC/'nonsevere_v1/PREDICTION_INPUTS.json'
N3 = C.ROOT/'_docs/experiments/pallet_n3_subpix_20261008_v1'
PAIRED = B/'raw_yolo_vs_N3_subpix_TEST49_paired_v1'
PROVENANCE = B/'yolo_N3_subpix_stage_provenance_review_v1/SOURCE_AND_CACHE_REVIEW_V1.json'
RAW, AFTER = 'NATIVE_BASE', 'INITIAL_N3_SUBPIX'
FIXED_IDS = (
 'eval_noapril:1775201414728017664','eval_noapril:1775201415399297536',
 'eval_noapril:1775201416944730624','eval_noapril:1775201418120676608',
 'eval_noapril:1775201421043401984','eval_noapril:1775201428535961600',
 'eval_noapril:1775201430786751744','eval_pallet07:1778652130452698368',
 'eval_pallet07:1778652132535256576','eval_pallet07:1778652138515809024')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
STAGES = ('q0','qN','qS','qFinal')
COLORS = {'q0':'#ff573d','qN':'#a020da','qS':'#007ac6','qFinal':'#00a56a'}
FONT = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')


def rel(path): return str(Path(path).resolve().relative_to(C.ROOT))


def paired_ids(pair):
    if pair['rows'] != 49 or pair['groups']['loss'] or tuple(pair['groups']['gain']) != FIXED_IDS:
        raise ValueError('Exact previously audited 10 gains / zero losses required')
    if len(pair['pairs']) != 49 or len(set(pair['ordered_ids'])) != 49:
        raise ValueError('All original TEST49 paired rows must remain')
    for row in pair['pairs']:
        a, b = row['raw_yolo_native_base'], row['N3_SUBPIX']
        expected = (not (a['translation_cm'] < 5 and a['rotation_deg'] < 5)
                    and b['translation_cm'] < 5 and b['rotation_deg'] < 5)
        if (row['id'] in FIXED_IDS) != expected:
            raise ValueError('Stored strict 5cm/5deg gain classification mismatch')
    return FIXED_IDS


def projection(pose, K, physical_dimensions):
    """Native corner channels use their own saved W/D and R_cf, never Rphysical."""
    if not pose.get('available'): raise ValueError('Both stored F poses must be available')
    R, Rp, t = (np.asarray(pose[k],float) for k in ('R_cf','R_physical','centroid'))
    dims = np.asarray(pose['cf_extents'],float); physical = np.asarray(physical_dimensions,float)
    if R.shape!=(3,3) or Rp.shape!=(3,3) or t.shape!=(3,) or dims.shape!=(3,):
        raise ValueError('Saved native F pose shape mismatch')
    if not all(np.isfinite(x).all() for x in (R,Rp,t,dims,physical)) or t[2]<=0:
        raise ValueError('Finite positive-depth cached pose required')
    for A in (R,Rp):
        if not np.allclose(A.T@A,np.eye(3),rtol=0,atol=1e-7) or abs(np.linalg.det(A)-1)>1e-7:
            raise ValueError('Proper stored rotation required')
    Q=np.eye(3) if abs(dims[0]-physical[0])<1e-6 else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    expected=physical if np.array_equal(Q,np.eye(3)) else physical[[2,1,0]]
    if not np.allclose(dims,expected,rtol=0,atol=1e-6) or not np.allclose(R@Q,Rp,rtol=0,atol=1e-7):
        raise ValueError('Saved native W/D / physical Q source=False convention mismatch')
    xyz=C.cuboid(*dims); camera=xyz@R.T+t
    if np.any(camera[:,2]<=0): raise ValueError('Stored wire behind camera')
    return C.project(xyz,R,t,K), Q


def stage_case(frame, stage, methods, metric):
    before=deepcopy((frame,stage,methods))
    if stage['id']!=frame['id'] or stage['method']!='N3_SUBPIX' or stage['raw_hw']!=frame['raw_hw']:
        raise ValueError('Exact original prediction id / camera canvas required')
    values={key:np.asarray(stage[key],float) for key in STAGES}
    if any(q.shape!=(9,2) for q in values.values()): raise ValueError('All four cached native9 stages required')
    for q,key in ((values['q0'],'BASE'),(values['qFinal'],'N3_SUBPIX')):
        if not np.array_equal(q,np.asarray(frame['points'][key],float),equal_nan=True):
            raise ValueError('Stage and frozen canonical coordinates differ')
    for method,key in ((RAW,'q0'),(AFTER,'qFinal')):
        row=methods[method]
        if row['id']!=frame['id'] or row['partition']!='TEST' or row['method']!=method:
            raise ValueError('Original TEST method identity mismatch')
        if not np.array_equal(np.asarray(row['native_points'],float),values[key],equal_nan=True):
            raise ValueError('Displayed points do not belong to stored F pose')
        if row['actual_pose']!=frame['initial_poses']['BASE' if method==RAW else 'N3_SUBPIX']:
            raise ValueError('Original F pose / canonical initial pose mismatch')
    projected={}; bases={}
    for method in (RAW,AFTER):
        projected[method],bases[method]=projection(methods[method]['actual_pose'],frame['K'],frame['xyz'])
    if C.finite(before)!=C.finite((frame,stage,methods)): raise AssertionError('Input mutation')
    return dict(id=frame['id'],image=frame['image'],image_sha256=frame['image_sha256'],
        raw_hw=deepcopy(frame['raw_hw']),K=deepcopy(frame['K']),physical_dimensions_xyz_m=deepcopy(frame['xyz']),
        prediction_support=deepcopy(frame['prediction_support']),stage_xy={k:v.tolist() for k,v in values.items()},
        poses={k:deepcopy(methods[k]['actual_pose']) for k in (RAW,AFTER)},
        native_model_projection_xy={k:v.tolist() for k,v in projected.items()},
        saved_physical_Q_verified={k:v.tolist() for k,v in bases.items()},metrics=deepcopy(metric),
        all_point_markers_are='historical detector/refined predictions; not observed points or GT',
        qN='N3 output already subject to its own original head cap',
        qS='SubPix output before total YOLO-anchor displacement cap; no stored standalone F pose',
        qFinal='original total YOLO-anchor cap output',new_F_or_NN_or_SubPix_or_reference_calls=0)


def viewport(case):
    h,w=case['raw_hw'];support=np.asarray(case['prediction_support'],bool)[:8]
    q=np.concatenate([np.asarray(case['stage_xy'][k],float)[:8][support] for k in STAGES])
    q=q[np.isfinite(q).all(1)&~np.all(q==-1,axis=1)]
    if not len(q): return [0.,float(w),0.,float(h)]
    lo=np.maximum(q.min(0)-50,[0,0]);hi=np.minimum(q.max(0)+50,[w,h])
    return [float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1])]


def _metadata(root_saved_render_GO):
    if root_saved_render_GO is not True: raise PermissionError('ROOT saved-render GO before any IO')
    if OUTPUT.exists(): raise FileExistsError('Preserve previous render attempts')
    seal=C.read(COMPARISON/'PREDICTION_SEAL.json');checks=C.read(N3/'CHECKS.json')
    if not seal['before_scoring'] or not seal['complete'] or seal['rows']!=665:
        raise ValueError('Original sealed 95 x 7 comparator required')
    if C.sha(COMPARISON/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256'] or C.sha(COMPARISON/'PROTOCOL.json')!=seal['protocol_sha256']:
        raise ValueError('Comparator bytes changed')
    if checks['status']!='PASS' or not checks['complete'] or C.sha(N3/'PREDICTIONS.jsonl.gz')!=checks['raw_rows_sha256']:
        raise ValueError('Original combined N3 journal changed')
    pairseal=C.read(PAIRED/'SEAL.json')
    pair=C.read(PAIRED/'PAIR_METRICS.json'); paired_ids(pair)
    # The small paired receipt is bound independently; no references are reopened.
    if C.sha(PAIRED/'PAIR_METRICS.json')!='5f77574acbc7605c639af1ac19ff75b903d8c6ff133fddfb9cd14632496f5854':
        raise ValueError('Audited paired metric bytes changed')
    frames={r['id']:r for r in C.read(COHORT)['frames'] if r['id'] in FIXED_IDS}
    if len(frames)!=10 or any(r['partition']!='TEST' for r in frames.values()): raise ValueError('Fixed TEST gain cohort mismatch')
    files=[COHORT,COMPARISON/'BASELINE_INPUTS.json',COMPARISON/'BASELINE_PROVENANCE.json',
        COMPARISON/'PROTOCOL.json',COMPARISON/'PREDICTION_SEAL.json',COMPARISON/'PREDICTIONS.jsonl.gz',
        N3/'PROTOCOL.json',N3/'CHECKS.json',N3/'PREDICTIONS.jsonl.gz',PAIRED/'PAIR_METRICS.json',PAIRED/'SEAL.json',PROVENANCE]
    for f in frames.values():
        path=C.ROOT/f['image']
        if C.sha(path)!=f['image_sha256']: raise ValueError('Original RGB file changed')
        files.append(path)
    sourcefiles=[Path(__file__),Path(C.__file__),Path(SELECT.__file__),
        C.ROOT/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
        C.ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py']
    protocol=C.read(COMPARISON/'PROTOCOL.json')
    for path in (COMPARISON/'BASELINE_INPUTS.json',sourcefiles[-2]):
        if C.sha(path)!=protocol['source_bindings'][str(path)]: raise ValueError('Original F basis source binding changed')
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'ANALYSIS_PLAN.json',dict(schema='SAVED_RAW_YOLO_N3_SUBPIX_GAIN10_PROCESS_PLAN_V1',
        UTC=datetime.now(timezone.utc).isoformat(),fixed_ids=list(FIXED_IDS),selection='10 gains and zero losses from sealed TEST49 strict<5cm and<5deg paired comparison; posthoc explanation only',
        source_sha256={rel(p):C.sha(p) for p in sourcefiles},input_sha256={rel(p):C.sha(p) for p in files},
        external_environment_binding={'font':{'path':str(FONT),'sha256':C.sha(FONT)}},
        reference_documents_opened=0,new_NN_PnP_SubPix_LM_or_measurement_calls=0,
        image_viewport='same per-case q-stage-only bounding box +50px clipped to actual canvas, never GT or pose error',
        point_semantics='all markers are historical predictions, no current S/U observation injection',
        stages=list(STAGES),only_stored_F_poses=[RAW,AFTER],missing_standalone_qS_F_pose_not_invented=True))
    C.write(OUTPUT/'PLAN_SEAL.json',dict(plan_sha256=C.sha(OUTPUT/'ANALYSIS_PLAN.json')))
    return frames,pair


def _saved_stages():
    result={}
    fields={'id','method','raw_hw',*STAGES}
    with gzip.open(N3/'PREDICTIONS.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            identity=SELECT.select_prediction_fields(line,{'id','method'})
            if identity.get('id') in FIXED_IDS and identity.get('method')=='N3_SUBPIX':
                if identity['id'] in result: raise ValueError('Duplicate cached combined stage')
                result[identity['id']]=SELECT.select_prediction_fields(line,fields)
    return result


def _saved_methods():
    result={};fields={'id','partition','method','actual_pose','native_points'}
    with gzip.open(COMPARISON/'PREDICTIONS.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            identity=SELECT.select_prediction_fields(line,{'id','method'})
            if identity.get('id') in FIXED_IDS and identity.get('method') in (RAW,AFTER):
                key=(identity['id'],identity['method'])
                if key in result: raise ValueError('Duplicate sealed comparison prediction')
                result[key]=SELECT.select_prediction_fields(line,fields)
    if len(result)!=20: raise ValueError('Both stored F poses for all ten cases required')
    return result


def _render(case,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager,patheffects
    from PIL import Image
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False})
    path=C.ROOT/case['image'];image=np.asarray(Image.open(path).convert('RGB'))
    if image.shape[:2]!=tuple(case['raw_hw']): raise ValueError('RGB canvas differs from frozen camera')
    folder.mkdir();shutil.copyfile(path,folder/'source_rgb.png')
    box=viewport(case);case['display_ROI_xyxy']=[box[0],box[2],box[1],box[3]]
    fig=plt.figure(figsize=(19,16),facecolor='white');grid=fig.add_gridspec(3,3,left=.035,right=.985,top=.92,bottom=.035,wspace=.15,hspace=.20)
    support=np.asarray(case['prediction_support'],bool)[:8]
    def panel(index,title,whole=False):
        inner=grid[index].subgridspec(2,1,height_ratios=(9,1.4),hspace=.04)
        ax=fig.add_subplot(inner[0]);note=fig.add_subplot(inner[1]);note.axis('off');ax.imshow(image)
        if not whole:ax.set_xlim(box[:2]);ax.set_ylim(box[3],box[2])
        ax.set_title(title,fontsize=12,pad=8);ax.set_xticks([]);ax.set_yticks([])
        return ax,note
    def points(ax,key,previous=None):
        q=np.asarray(case['stage_xy'][key],float)
        for k in range(8):
            if not support[k] or not np.isfinite(q[k]).all() or np.all(q[k]==-1): continue
            if previous:
                old=np.asarray(case['stage_xy'][previous],float)[k]
                if np.isfinite(old).all() and not np.all(old==-1):
                    ax.scatter(*old,marker='x',color='#666666',s=28,zorder=3)
                    ax.annotate('',q[k],old,arrowprops={'arrowstyle':'->','color':COLORS[key],'lw':1.8},zorder=4)
            ax.scatter(*q[k],s=30,facecolor='none',edgecolor=COLORS[key],lw=1.6,zorder=5)
            text=ax.annotate(str(k),q[k],xytext=(5,5 if k%2 else -12),textcoords='offset points',color=COLORS[key],fontsize=10,weight='bold',zorder=6)
            text.set_path_effects([patheffects.withStroke(linewidth=2,foreground='white')])
    def pose(ax,method,key):
        p=np.asarray(case['native_model_projection_xy'][method]);color=COLORS[key]
        for a,b in EDGES:ax.plot(p[[a,b],0],p[[a,b],1],color=color,lw=1.8,alpha=.9)
        points(ax,key)
    def note(ax,text):ax.text(.01,.98,text,va='top',fontsize=10,transform=ax.transAxes,wrap=True)
    ax,n=panel(0,'원본 사진 · 전체 프레임',True);note(n,'같은 실제 사진. 아래 확대 범위는 저장된 2D 예측 좌표로만 정함.')
    ax,n=panel(1,'안 쓴 경로 ① YOLO q0');points(ax,'q0');note(n,'원 YOLO 코너 예측. 점 번호는 원 native 채널 0–7. 관측 정답 아님.')
    ax,n=panel(2,'안 쓴 경로 ② 같은 F의 저장 자세');pose(ax,RAW,'q0');m=case['metrics']['raw_yolo_native_base'];note(n,f"T {m['translation_cm']:.3f} cm · R {m['rotation_deg']:.3f}° · 기준 실패\n선=저장된 F 예측, 점=입력 q0")
    ax,n=panel(3,'쓴 경로 ① 같은 YOLO q0');points(ax,'q0');note(n,'같은 시작 좌표. 이 비교에 최신 상태 보정·마스크는 넣지 않음.')
    ax,n=panel(4,'쓴 경로 ② 저장된 N3 qN');points(ax,'qN','q0');note(n,'회색 × q0 → 보라 ○ qN. qN은 원 N3 자체 이동한도 적용 후 출력.')
    ax,n=panel(5,'쓴 경로 ③ 저장된 SubPix qS');points(ax,'qS','qN');note(n,'회색 × qN → 파랑 ○ qS. 총 이동한도 전 2D 좌표만; 별도 자세 없음.')
    ax,n=panel(6,'쓴 경로 ④ 총 이동한도 qFinal');points(ax,'qFinal','qS');note(n,'회색 × qS → 초록 ○ qFinal. YOLO q0 기준 원 총이동한도 적용.')
    ax,n=panel(7,'쓴 경로 ⑤ 같은 F의 저장 자세');pose(ax,AFTER,'qFinal');m=case['metrics']['N3_SUBPIX'];note(n,f"T {m['translation_cm']:.3f} cm · R {m['rotation_deg']:.3f}° · 기준 성공\n선=저장된 F 예측, 점=입력 qFinal")
    ax=fig.add_subplot(grid[8]);ax.axis('off')
    ax.text(.02,.98,'두 경로의 차이',va='top',fontsize=15,weight='bold')
    ax.text(.02,.84,'안 씀\nYOLO q0 → F\n\n씀\nYOLO q0 → N3 → SubPix\n→ q0 기준 총 이동한도 → 같은 F\n\nF: W/D 선택 → SQPnP + LM\n모두 당시 저장 결과; 이번에 다시 풀지 않음.\n\n기준: T < 5 cm 그리고 R < 5°\n저장 평가 숫자만 재표시; GT wire 없음.\n점은 모두 예측이며 S/U 관측점으로 쓰지 않음.\n궤적 화살표는 2D 단계 차이; LM 경로 아님.',va='top',fontsize=12,linespacing=1.45)
    fig.suptitle(case['id']+'\n실제 저장 단계 비교 · YOLO만 쓴 경우 vs N3+SubPix를 쓴 경우',fontsize=19)
    fig.savefig(folder/'process.png',dpi=120,facecolor='white');plt.close(fig)
    C.write(folder/'PROCESS_DATA.json',case)


def main(*,root_saved_render_GO=False):
    frames,pair=_metadata(root_saved_render_GO)
    stages=_saved_stages();methods=_saved_methods();metrics={r['id']:r for r in pair['pairs']}
    baseline={x['frame']['id']:x['frame'] for x in C.read(COMPARISON/'BASELINE_INPUTS.json')['frames']}
    cards=[];assets={};md=['# YOLO만 쓴 경우와 N3+SubPix를 쓴 경우: 달라진 10장','',
        '기존 TEST49 저장 평가에서 성공은 30/49 → 40/49다. 정확히 10장이 실패에서 성공으로 바뀌었고 역방향 손실은 0장이다. 기준은 기존의 **T < 5 cm 및 R < 5°**이며, 이 갤러리는 그 10장의 사후 설명이다.','',
        '각 그림의 윗줄은 원 YOLO 좌표 → 같은 F의 저장 자세, 아랫줄은 YOLO → N3(자체 이동한도 후) → SubPix → YOLO 기준 총 이동한도 → 같은 F의 저장 자세다. 모든 점은 당시 저장된 예측이다. SubPix 직후의 별도 자세나 LM 중간 궤적은 저장되지 않아 만들지 않았다.','',
        'F는 동일한 예측 기반 W/D 선택 및 SQPnP+LM이다. 이번에는 사진과 저장 좌표·자세를 그렸으며 새 추론·SubPix·PnP·정답 조회는 0회다. 정답 wire/정답 기반 ROI도 없다.','']
    for i,identifier in enumerate(FIXED_IDS):
        frame=frames[identifier]
        for key in ('points','initial_poses','image_sha256','K','xyz','raw_hw'):
            if frame[key]!=baseline[identifier][key]:raise ValueError('Canonical and sealed baseline input mismatch '+key)
        case=stage_case(frame,stages[identifier],{m:methods[(identifier,m)] for m in (RAW,AFTER)},metrics[identifier])
        folder=OUTPUT/f'case{i:02d}';_render(case,folder)
        cards.append(dict(id=identifier,folder=folder.name,metrics=metrics[identifier]))
        md+=['## '+identifier,'',f'[원본 전체 사진]({folder.name}/source_rgb.png) · [전체 과정 확대 PNG]({folder.name}/process.png)','',f'![저장 단계 비교]({folder.name}/process.png)','']
    (OUTPUT/'REPORT_KO.md').write_text('\n'.join(md),encoding='utf8')
    page=['<!doctype html><meta charset="utf-8"><title>달라진 10장: YOLO / N3+SubPix</title><style>body{font-family:sans-serif;max-width:1400px;margin:30px auto;padding:20px;background:#f4f6f8;color:#182330}section{background:white;padding:22px;margin:25px 0;border-radius:12px}img{width:100%;height:auto}nav a{margin:8px;display:inline-block}p{line-height:1.7}</style>',
        '<h1>YOLO만 쓴 경우와 N3+SubPix를 쓴 경우: 달라진 10장</h1><p>기존 TEST49: 성공 30/49 → 40/49. 실패→성공 10장, 성공→실패 0장. 기준 T &lt; 5 cm 및 R &lt; 5°. 각 사례는 같은 사진·같은 확대 범위로 비교합니다.</p><p>윗줄: YOLO → 같은 F. 아랫줄: YOLO → N3 → SubPix → 총 이동한도 → 같은 F. 모든 점과 선은 저장된 <b>예측</b>이며 정답이 아닙니다. 중간 좌표는 실제 캐시, 마지막 선은 저장된 F 자세입니다. 새 PnP/추론/정답 조회는 없습니다.</p><nav>']
    page += [f'<a href="#c{i}">{i+1}. {html.escape(c["id"])}</a>' for i,c in enumerate(cards)]
    page+=['</nav>']
    for i,c in enumerate(cards):
        a,b=c['metrics']['raw_yolo_native_base'],c['metrics']['N3_SUBPIX']
        page.append(f'<section id="c{i}"><h2>{i+1}. {html.escape(c["id"])}</h2><p>YOLO+F: T {a["translation_cm"]:.3f} cm / R {a["rotation_deg"]:.3f}° → N3+SubPix+F: T {b["translation_cm"]:.3f} cm / R {b["rotation_deg"]:.3f}°</p><p><a href="{c["folder"]}/source_rgb.png">원본 전체 사진</a> · <a href="{c["folder"]}/process.png">과정 PNG 원본 크기</a> · <a href="{c["folder"]}/PROCESS_DATA.json">저장 값·출처</a></p><a href="{c["folder"]}/process.png"><img loading="lazy" src="{c["folder"]}/process.png" alt="YOLO와 N3 SubPix 저장 과정 비교"></a></section>')
    (OUTPUT/'index.html').write_text('\n'.join(page),encoding='utf8')
    for p in OUTPUT.rglob('*'):
        if p.is_file():assets[str(p.relative_to(OUTPUT))]=C.sha(p)
    C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='SAVED_RAW_YOLO_N3_SUBPIX_GAIN10_RENDER_SEAL_V1',
        cases=cards,rows=10,stage_arrays40=True,stored_F_poses20=True,plan_sha256=C.sha(OUTPUT/'ANALYSIS_PLAN.json'),
        assets=assets,new_NN_SubPix_PnP_optimizer_reference_reads=0,visual_QA_pending=True))
    print(C.sha(OUTPUT/'RENDER_SEAL.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root-saved-render-GO',action='store_true')
    main(root_saved_render_GO=parser.parse_args().root_saved_render_GO)
