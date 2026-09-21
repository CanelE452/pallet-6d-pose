"""Same 18 CAD frames, frozen R0/N3/Replay; no training or split changes."""
from pathlib import Path
import argparse
import copy
import hashlib
import html
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/research/pallet_dim_conditioned_p_v1'))
from scripts.research.pallet_cad_r0_gt_gallery_v1 import build as G
from inference import preservation
import eval_math
import pose

OUT = ROOT/'outputs/pallet_cad_refiner_comparison_v1'
N3 = ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV'
RDOC = ROOT/'_docs/experiments/pallet_posefix_replay_v1'
SPLIT = ROOT/'_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json'
ARMS = ['R0','N3_DIM_SYM_seed1','N3_DIM_SYM_seed2','N3_DIM_SYM_seed3','REPLAY_RAW','REPLAY_CAP1PCT']
DISPLAY = ['R0','N3_DIM_SYM_seed1','REPLAY_RAW']
LABEL = dict(R0='R0 · 보정 없음',N3_DIM_SYM_seed1='DIM-SYM · seed1',N3_DIM_SYM_seed2='DIM-SYM · seed2',
             N3_DIM_SYM_seed3='DIM-SYM · seed3',REPLAY_RAW='Replay PoseFix · raw',REPLAY_CAP1PCT='Replay PoseFix · 기존 1% cap')
read, bind, verify = G.read, G.binding, G.verify


def write(name, payload):
    G.write(OUT/name, payload)


def top(p):
    return p['candidates'][p['selected_index']]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    records = [r for r in read(G.PROTOCOL)['records'] if r['session']=='eval_cad']
    base_payload = read(G.PRED)
    base = {r['id']:r['prediction'] for r in base_payload['records'] if r['id'].startswith('eval_cad:')}
    assert len(records)==len(base)==18
    fit = read(RDOC/'FIT.json'); verify(fit['checkpoint']);verify(base_payload['checkpoint'])
    sources = [G.PROTOCOL,G.PRED,G.SCORES,RDOC/'FIT.json',RDOC/'PROTOCOL.json',RDOC/'INPUT_LOCK.json',SPLIT,Path(__file__)]
    sources += [ROOT/fit['checkpoint']['path'],ROOT/base_payload['checkpoint']['path']]
    predictions={'R0':base}
    for name in ARMS[1:4]:
        path=N3/f'{name}.json';p=read(path);assert p['complete'] and not p['GT_input'];verify(p['checkpoint'])
        rows={r['id']:r for r in p['records']}
        predictions[name]={r['id']:rows[r['id']] for r in records}
        sources += [path,ROOT/p['checkpoint']['path']]
    for r in records:
        for field in ('image','annotation'):
            verify(r[field]);sources.append(ROOT/r[field]['path'])
        for arm in predictions:
            p=predictions[arm][r['id']]
            assert p['selected_index']==base[r['id']]['selected_index']
            preservation(base[r['id']]['candidates'],p['candidates'],p['selected_index'])
    split=read(SPLIT);train=split['train']
    actual_train_ids=set(read(RDOC/'INPUT_LOCK.json')['real_ids'])
    assert actual_train_ids=={r['id'] for r in train}
    assert not {r['id'] for r in records}&actual_train_ids
    assert not {r['image']['sha256'] for r in records}&{r['image']['sha256'] for r in train}
    assert not {'eval_cad'}&{r['session'] for r in train}
    write('PROTOCOL.json',dict(frames=18,ids=[r['id'] for r in records],arms=ARMS,display=DISPLAY,
        visual_seed='Fixed seed1 for illustration, not chosen by these scores; seeds2/3 and arithmetic mean also reported',
        methods='Each refiner receives the same R0 points. Not sequential. Replay raw is main; historical 1% cap is secondary.',
        score9='Same fixed-index supervised corner8+center as preceding CAD report',
        score8='Original eval_math whole-object C2 symmetry-aware corner8; no coordinates changed by evaluation matching',
        pose='Same frozen prediction-only selector + SQPnP/LM; reconstructed reference, not external physical ground truth',
        reused_DEV=True,independent_confirmation=False,training_updates=0,split_changed=False,
        actual_replay_train_id_hash_session_overlap=False,
        reservation_caveat='REC_021 was previously reserved from DEV72 even though the actual TRAIN9 contain no CAD. This new requested CAD diagnostic is not the old preregistered DEV72 test.',
        sources=[bind(p) for p in sorted(set(sources))]))
    # Inference file deliberately excludes target coordinates and annotations.
    write('INPUTS.json',[dict(id=r['id'],image=r['image'],prediction=base[r['id']]) for r in records])
    write('CACHED_PREDICTIONS.json',predictions)
    print('PREPARED 18 frames; frozen N3 seeds 1/2/3 and Replay last300',flush=True)


def check():
    for b in read(OUT/'PROTOCOL.json')['sources']:
        amendment=OUT/'SCORER_ADAPTER_FIX.json'
        if b['path']==str(Path(__file__).relative_to(ROOT)) and amendment.exists():
            fix=read(amendment);assert b['sha256']==fix['original_source_sha256']
            verify(fix['corrected_source'])
        else:verify(b)


def infer():
    import torch
    from scripts.research.pallet_posefix_replay_v1 import core as N
    check();N.setup();gpu_before=N.E.gpu();assert torch.cuda.is_available()
    if (OUT/'REPLAY_PREDICTIONS.json').exists():
        print('REPLAY ALREADY EXISTS');return
    fit=read(RDOC/'FIT.json');ck=torch.load(ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['step']==300 and ck['protocol_sha256']==bind(RDOC/'PROTOCOL.json')['sha256']
    model=N.C.PoseFixPallet9();model.load_state_dict(ck['model_state_dict'],strict=True)
    model=model.cuda().eval().requires_grad_(False);del ck
    result={a:{} for a in ARMS[-2:]}
    # One old DEV72 sample is a numerical parity check, not a new evaluation frame.
    historical=read(ROOT/'data/pallet/results/pallet_posefix_replay_v1/PREDICTIONS.json')['DEV72'][0]
    verify(historical['image'])
    with torch.inference_mode():
        q=N.C.predict(model,cv2.imread(str(ROOT/historical['image']['path'])),historical['predictions']['R0'],None)
        diff=float(np.max(np.abs(np.asarray(top(q)['keypoints_xy'])-np.asarray(top(historical['predictions']['POSEFIX_RAW'])['keypoints_xy']))))
        assert diff<=.01, ('Historical 0.01px replay-parity tolerance exceeded',diff)
        print('OLD_REPLAY_PARITY max_abs_px',diff,flush=True)
        for i,r in enumerate(read(OUT/'INPUTS.json'),1):
            verify(r['image']);im=cv2.imread(str(ROOT/r['image']['path']));assert im.shape[:2]==(480,640)
            base=r['prediction'];before=copy.deepcopy(base)
            raw=N.C.predict(model,im,base,None);assert base==before
            capped=N.C.cap_prediction(base,raw,.01,im.shape[:2])
            for arm,p in [('REPLAY_RAW',raw),('REPLAY_CAP1PCT',capped)]:
                assert p['selected_index']==base['selected_index']
                preservation(base['candidates'],p['candidates'],p['selected_index'])
                result[arm][r['id']]=p
            if i%6==0:print('CAD_REPLAY',i,18,N.E.gpu(),flush=True)
    gpu_after=N.E.gpu();del model;torch.cuda.empty_cache()
    write('REPLAY_PREDICTIONS.json',dict(predictions=result,checkpoint=fit['checkpoint'],GT_input=False,
        frozen=True,training_updates=0,old_DEV72_parity_max_abs_px=diff,gpu_before=gpu_before,gpu_after=gpu_after))
    check();print('REPLAY_COMPLETE',flush=True)


def basic_summary(rows):
    e=np.concatenate([r['supervised_errors'] for r in rows])
    return dict(frames=len(rows),points=len(e),median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)),
        PCK10=float(np.mean(e<=10)),PCK20=float(np.mean(e<=20)),max_px=float(e.max()),
        all20_frames=sum(r['max_px']<=20 for r in rows))


def pose_summary(rows):
    valid=[r for r in rows if r['available']]
    result=dict(coverage=len(valid)/len(rows),ADDsym_AUC=pose.pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in rows],1.))
    for k in ['rotation_deg','yaw_deg','translation_cm','IoU3D']:
        result[k]=float(np.median([r[k] for r in valid])) if valid else None
    return result


def score():
    check();cv2.setNumThreads(1)
    predictions=read(OUT/'CACHED_PREDICTIONS.json')
    predictions.update(read(OUT/'REPLAY_PREDICTIONS.json')['predictions'])
    meta,truth=pose.metadata('REAL_DEV')
    records=[r for r in read(G.PROTOCOL)['records'] if r['session']=='eval_cad']
    # PnP solves are fixed before annotation-based score calculation.
    poses={arm:{r['id']:pose.infer(top(predictions[arm][r['id']])['keypoints_xy'],*meta[r['id']]) for r in records} for arm in ARMS}
    write('POSE_PREDICTIONS.json',poses)
    groups=read(ROOT/'_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    perms=next(g['permutations'] for g in groups if g['object_type']=='plastic_standard_110x130x11')
    rows={a:[] for a in ARMS};rows8={a:[] for a in ARMS};rows6={a:[] for a in ARMS}
    for r in records:
        obj=read(ROOT/r['annotation']['path'])['objects'][0];ann=obj['keypoint_annotations']
        gt=np.asarray([p['xy'] for p in ann],float);valid=np.asarray([p['visibility']>0 and p.get('in_frame',True) for p in ann])
        original=np.asarray(top(predictions['R0'][r['id']])['keypoints_xy'])
        for arm in ARMS:
            pred=np.asarray(top(predictions[arm][r['id']])['keypoints_xy']);errors=np.linalg.norm(pred-gt,axis=1)
            movement=np.linalg.norm(pred[:8]-original[:8],axis=1)
            rows[arm].append(dict(id=r['id'],gt=gt.tolist(),prediction=pred.tolist(),valid=valid.tolist(),
                errors_px=errors.tolist(),supervised_errors=errors[valid].tolist(),median_px=float(np.median(errors[valid])),
                mean_px=float(errors[valid].mean()),max_px=float(errors[valid].max()),movement_px=movement.tolist(),
                truncated=bool(obj.get('truncation',{}).get('is_truncated'))))
            m=eval_math.measure(pred,gt,valid,perms,(480,640),True,True);m.update(id=r['id'],session='eval_cad');rows8[arm].append(m)
            rows6[arm].append(pose.metric((r['id'],poses[arm][r['id']],truth[r['id']])))
    sums={}
    for arm in ARMS:
        delta=np.asarray([n['mean_px']-b['mean_px'] for b,n in zip(rows['R0'],rows[arm])])
        moves=np.concatenate([r['movement_px'] for r in rows[arm]])
        sums[arm]=dict(fixed9=basic_summary(rows[arm]),sym_corner8=eval_math.summary(rows8[arm]),pose=pose_summary(rows6[arm]),
            movement=dict(median_px=float(np.median(moves)),max_px=float(moves.max())),
            improved_frames=int((delta < -1e-9).sum()),worsened_frames=int((delta > 1e-9).sum()))
    # Reproduce the exact preceding R0 9-point report on the same supervised set.
    old={r['frame_id']:r for r in read(G.SCORES)['records']}
    for r in rows['R0']:np.testing.assert_allclose(r['supervised_errors'],old[r['id']]['errors_px'],rtol=0,atol=1e-9)
    baseline6={r['id']:r for r in read(ROOT/'_docs/experiments/pallet_final_paper_tables_v1/R0_POSE.json')['metrics']}
    for r in rows6['R0']:
        for k in ['translation_cm','rotation_deg','yaw_deg','IoU3D','ADDsym_normalized']:
            assert abs(r[k]-baseline6[r['id']][k])<1e-5,(r['id'],k)
    mean={k:float(np.mean([sums[a]['fixed9'][k] for a in ARMS[1:4]])) for k in sums[ARMS[1]]['fixed9']}
    write('METRICS.json',dict(summary=sums,N3_three_seed_mean_fixed9=mean,rows=rows,rows_corner8=rows8,rows_pose=rows6,
        R0_prior_score_parity=True,detector_and_center_preserved=True,AP_change=0,
        AP_note='All detector outputs identical; no claim of new AP measurement on positive-only 18 frames',
        no_training=True,no_split_change=True,GT_for_scoring_only=True))
    print(json.dumps(sums,ensure_ascii=False),flush=True)


def render_panel(rgb,row,title):
    canvas=rgb.copy();gt=np.asarray(row['gt']);pred=np.asarray(row['prediction'])
    G.draw(canvas,gt,G.GREEN);G.draw(canvas,pred,G.BLUE,'P',True)
    head=np.full((64,640,3),(30,25,19),np.uint8)
    G.text(head,title,(12,21));G.text(head,f'med {row["median_px"]:.2f}px | max {row["max_px"]:.2f}px | move max {max(row["movement_px"]):.2f}px',(12,46))
    return np.concatenate([head,canvas],axis=0)


def gallery():
    result=read(OUT/'METRICS.json');rows={a:{r['id']:r for r in rs} for a,rs in result['rows'].items()}
    records=[r for r in read(G.PROTOCOL)['records'] if r['session']=='eval_cad'];cards=[]
    table=[];table6=[]
    for arm in ARMS:
        s=result['summary'][arm];t=s['fixed9'];p=s['pose']
        table.append(f'<tr><td>{LABEL[arm]}</td><td>{t["median_px"]:.2f}</td><td>{t["p90_px"]:.2f}</td><td>{100*t["PCK20"]:.1f}%</td><td>{t["all20_frames"]}/18</td><td>{s["improved_frames"]} / {s["worsened_frames"]}</td></tr>')
        table6.append(f'<tr><td>{LABEL[arm]}</td><td>{p["rotation_deg"]:.3f}</td><td>{p["translation_cm"]:.3f}</td><td>{p["IoU3D"]:.4f}</td><td>{p["ADDsym_AUC"]:.4f}</td></tr>')
    for i,r in enumerate(records,1):
        im=cv2.imread(str(ROOT/r['image']['path']));panels=[]
        for arm in ARMS:
            panel=render_panel(im,rows[arm][r['id']],f'{i:02d}/18  {arm}')
            G.image_write(OUT/f'{i:02d}_{arm}.png',panel)
            if arm in DISPLAY:panels.append(panel)
        # 2x2 keeps each image readable: GT, raw R0, N3, Replay.
        gt_panel=im.copy();G.draw(gt_panel,np.asarray(rows['R0'][r['id']]['gt']),G.GREEN,'G')
        head=np.full((64,640,3),(30,25,19),np.uint8);G.text(head,f'{i:02d}/18  GT / existing annotation',(12,23))
        G.text(head,'All output coordinates unchanged; GT for display only',(12,47))
        gt_panel=np.concatenate([head,gt_panel])
        grid=np.concatenate([np.concatenate([gt_panel,panels[0]],axis=1),np.concatenate(panels[1:],axis=1)],axis=0)
        file=OUT/f'{i:02d}_comparison.png';G.image_write(file,grid)
        items=[]
        for a in DISPLAY:
            q=rows[a][r['id']];d=q['mean_px']-rows['R0'][r['id']]['mean_px']
            items.append(f'{LABEL[a]}: 중앙 {q["median_px"]:.2f}px / 최대 {q["max_px"]:.2f}px / 프레임 평균 변화 {d:+.2f}px')
        extras=' · '.join(f'<a href="{i:02d}_{a}.png" target="_blank">{LABEL[a]} 확대</a>' for a in ARMS)
        cards.append(f'<section id="f{i}" class="card"><h2>{i:02d}/18 · {html.escape(r["id"])}</h2><p>'+(' · '.join(items))+f'</p><a href="{file.name}" target="_blank"><img src="{file.name}" width="1280" height="1088"></a><p>{extras}</p></section>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CAD 18장 · R0 / DIM-SYM / Replay 보정 비교</title>
<style>body{background:#111b23;color:#eff4fa;margin:0;font:16px/1.65 system-ui}main{max-width:1360px;padding:22px;margin:auto}h1{font-size:26px}h2{font-size:18px}a{color:#8bc9ff}img{width:100%;height:auto}.card{padding:24px 0;border-top:1px solid #475764}table{border-collapse:collapse}td,th{padding:8px 14px;border-bottom:1px solid #485761;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.sticky{position:sticky;top:0;background:#111b23f5;padding:10px;text-align:center;z-index:1}.note{background:#293441;padding:14px}.green{color:#55f546}.blue{color:#23aaff}</style>
<div class="sticky">위: 정답 / R0 · 아래: DIM-SYM seed1 / Replay raw · <span class="green">초록 = 정답</span> / <span class="blue">파랑 = 각 모델 출력</span></div><main><h1>eval_cad 전체 18장 — 보정 전·후</h1>
<p>같은 R0 출력에 각 보정기를 별도로 적용했습니다. 보정기 연쇄 적용, 필터, 좌표 재배열, 재학습, train/eval 변경은 없습니다. Self-training 학생은 이 비교에 없습니다.</p>
<p class="note">DIM-SYM은 기존 N3 seed1을 고정 대표로 그렸으며 seed2/3도 모두 보고합니다. Replay는 기존 실사9장+합성 replay 학습의 last300 모델입니다.
CAD는 실제 TRAIN9와 이미지·세션이 겹치지 않지만, 과거 후보 촬영으로 예약되어 DEV72에서는 제외됐던 세션입니다. 이번 요청에 따른 반복 DEV 진단이며 독립 최종 검증은 아닙니다.</p>
<h2>같은 18장 지표 — 앞선 보고와 동일한 코너8+중심점 정의</h2><p>평가 가능한 154개 점, 잘림 4장도 그대로 포함. 개선/악화는 이미지별 평균 점 오차의 R0 대비 변화입니다.</p>
<div class="scroll"><table><tr><th>모델</th><th>중앙 px ↓</th><th>P90 px ↓</th><th>20px 이내 점 ↑</th><th>모든 점 ≤20px ↑</th><th>개선 / 악화 이미지</th></tr>'''+''.join(table)+'''</table></div>
<h2>6D — 같은 PnP와 같은 복원 참조</h2><p>외부 장비로 측정한 GT가 아니라 기존 2D 정답·카메라·치수에서 기하적으로 복원한 참조입니다. GT는 추론 입력이 아닙니다.</p>
<div class="scroll"><table><tr><th>모델</th><th>회전 중앙 ° ↓</th><th>이동 중앙 cm ↓</th><th>IoU3D 중앙 ↑</th><th>ADDsym AUC ↑</th></tr>'''+''.join(table6)+'''</table></div>
<p>검출 박스·점수·후보·중심점은 전부 동일하므로 검출 AP는 보정으로 변하지 않습니다. 이 18장만의 AP를 새로 측정했다고 주장하지 않습니다.
논문용 C2 대칭 코너8 지표와 원시 좌표는 <a href="METRICS.json">METRICS.json</a>에 별도로 보존했습니다.</p>
<nav>'''+ ' · '.join(f'<a href="#f{i}">{i:02d}</a>' for i in range(1,19))+'</nav>'+''.join(cards)+'</main></html>'
    write('index.html',page)
    print('HTML',OUT/'index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','infer','score','gallery'])
    globals()[parser.parse_args().phase]()
