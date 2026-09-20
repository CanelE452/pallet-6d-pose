"""Read-only scientific audit and complete gallery after all outputs are locked."""
import json
import subprocess
import sys
import cv2
import numpy as np
import torch
from . import heatmap_mode_recovery as H
from .recovery_damage import draw

C=H.C
OUT=C.OUT/'selftrain_recovery_v1/heatmap_mode_recovery'


def main():
    H.verify();H.N.setup()
    run=subprocess.run([sys.executable,'-m','pytest','-q',str(C.ROOT/'scripts/research/pallet_type_selftrain_v1/test_heatmap_mode_decoder.py')],capture_output=True,text=True)
    print(run.stdout,flush=True);assert run.returncode==0,run.stderr
    assert '7 passed' in run.stdout
    for b in C.read(H.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    records=C.read(H.DOC/'EVAL_PROTOCOL.json')['records']
    metrics={a:{r['id']:r for r in C.read(H.RAW/f'SCREEN_{a}.json')['metrics']} for a in H.ARMS}
    metrics['R0']={r['id']:r for r in C.read(H.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    preds={a:{r['id']:r['prediction'] for r in C.read(H.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in H.ARMS}
    preds['R0']={r['id']:r['prediction'] for r in C.read(H.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    pe,pop=H.R.E.O.population_metadata()
    targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in metrics['R0']}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    cache={m:np.load(H.RAW/f'REAL_LOGITS_{m}.npz') for m in H.MODELS}
    redecoded={m:H.D.ambiguity(torch.from_numpy(z['logits'])) for m,z in cache.items()}
    parity={a:0. for a in H.ARMS};cards=[];checks=0;recoveries=[]
    (OUT/'images').mkdir(parents=True,exist_ok=True)
    for idx,r in enumerate(records):
        key=r['id'];t=targets[key];b=metrics['R0'][key]
        C.verify(r['image']);C.verify(r['annotation']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        item=H.N.C.prepare_input(im,preds['R0'][key])
        for name in H.MODELS:
            assert cache[name]['ids'][idx]==key
            np.testing.assert_array_equal(cache[name]['matrix'][idx],item['matrix'])
            for decoder in ['MEAN','MODE']:
                arm=f'{name}_{decoder}'
                out=H.replace(preds['R0'][key],item,redecoded[name][decoder.lower()][idx].numpy())
                expected=np.array(H.P.top(preds[arm][key])['keypoints_xy'])
                actual=np.array(H.P.top(out)['keypoints_xy'])
                parity[arm]=max(parity[arm],float(np.abs(actual-expected).max()))
                np.testing.assert_allclose(actual,expected,atol=.003,rtol=0)
                H.P.assert_preserved(preds['R0'][key],preds[arm][key]);checks+=1
        counts={}
        for arm in H.ARMS:
            n=metrics[arm][key]
            assert n['matched']==b['matched'] and n['detected']==b['detected']
            pairs=list(zip(b['canonical_errors'],n['canonical_errors']))
            recovered=[j for j,(x,y) in enumerate(pairs) if x is not None and x>20 and y<=10] if b['matched'] else []
            hard40=[j for j,(x,y) in enumerate(pairs) if x is not None and x>40 and y<=10] if b['matched'] else []
            damaged=[j for j,(x,y) in enumerate(pairs) if x is not None and x<5 and y>10] if b['matched'] else []
            counts[arm]=dict(recovered=recovered,hard40=hard40,damaged=damaged)
            for j in hard40:
                recoveries.append(dict(id=key,arm=arm,canonical_corner=j,R0_error=b['canonical_errors'][j],new_error=n['canonical_errors'][j],
                    R0_branch=b['branch'],new_branch=n['branch'],same_branch=b['branch']==n['branch'],image_index=idx))
        panels=[]
        for arm in ['R0','BASE_MEAN','BASE_MODE','R0','DENOISE_MEAN','DENOISE_MODE']:
            n=metrics[arm][key]
            panels.append(draw(im,H.P.top(preds[arm][key])['keypoints_xy'],np.asarray(t.keypoints_xy),np.asarray(t.keypoint_supervision_mask,bool),perms[n['branch']],arm+' / NOT DEPLOYED' if arm!='R0' else 'R0',n['frame_mean_px']))
        tile=np.concatenate([np.concatenate(panels[:3],axis=1),np.concatenate(panels[3:],axis=1)],axis=0)
        name=f'images/{idx:03d}.jpg';dest=OUT/name
        if not dest.exists():assert cv2.imwrite(str(dest),tile,[cv2.IMWRITE_JPEG_QUALITY,88])
        cards.append(dict(id=key,image=name,counts=counts,mean={a:metrics[a][key]['frame_mean_px'] for a in ['R0',*H.ARMS]}))
    results=C.read(H.DOC/'RESULTS.json')
    for arm in H.ARMS:
        d=results['results']['full194'][arm]
        assert sum(len(r['counts'][arm]['recovered']) for r in cards)==d['recovery']['recovered']
        assert sum(len(r['counts'][arm]['damaged']) for r in cards)==d['recovery']['damaged']
        assert sum(len(r['counts'][arm]['hard40']) for r in cards)==d['tails']['40']['recovered']
    assert all(r['same_branch'] for r in recoveries),'Inspect correspondence changes before claiming pure location recovery'
    C.freeze(OUT/'CASES.json',cards)
    payload=json.dumps(cards,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>큰 코너 위치 오차: 전체 평균과 확률 봉우리 비교</title>
<style>body{background:#101820;color:#eee;font:17px sans-serif;max-width:1800px;margin:24px auto;padding:16px}article{border:1px solid #789;padding:12px;margin:24px 0}img{width:100%}select,input{font:inherit;padding:8px}.warning{background:#682f27;padding:16px}</style>
<h1>실제 위치 보정 — 기존 확률 분포를 읽는 방식 비교</h1><p class="warning">미채택 후보입니다. 일부40px 이상 오차를 복구했지만 정상 코너를 손상시켜 안전 기준에 실패했습니다. 기존 최종 모델과 수도레이블은 바꾸지 않았습니다.</p>
<p>위: R0 / 기존 Replay 전체 평균 / 같은 Replay의 최상위 봉우리 주변 평균. 아래: R0 / 기존 수도레이블 denoise 모델 전체 평균 / 같은 모델의 봉우리 주변 평균. 추가 학습·태그·수동 레이블은 없습니다. 모든 예측을 고정한 뒤에만 정답으로 채점·표시했습니다.</p>
<p>노랑=예측, 자홍 X=기존 평가 참조점. 공식 C2 분기 유지. 반복 DEV194장이고 역사적 teacher 학습3장 overlap이 있습니다. 아래 성공 사진만으로 성능 개선을 주장하지 않습니다.</p>
<select id="model"><option value="BASE_MODE">기존 Replay 봉우리 좌표</option><option value="DENOISE_MODE">DENOISE 봉우리 좌표</option></select>
<select id="filter"><option value="hard40">40px 이상 → 10px 이내 실제 복구</option><option value="damage">정상 코너 손상</option><option value="all">전체194장</option><option value="example">사용자의 의자·콘 예시</option></select><p id="count"></p><main></main>
<script>const data=__DATA__;const el=id=>document.getElementById(id);function render(){let f=el('filter').value,m=el('model').value,rows=data.filter(r=>f==='all'||(f==='example'?r.id==='eval_pallet07:1778652166837872128':f==='hard40'?r.counts[m].hard40.length:r.counts[m].damaged.length));el('count').textContent=rows.length+' /194장';const root=document.querySelector('main');root.replaceChildren();for(const r of rows){const a=document.createElement('article'),h=document.createElement('h3'),im=document.createElement('img');h.textContent=r.id+' | '+m+' 40px복구 '+r.counts[m].hard40.length+' / 정상손상 '+r.counts[m].damaged.length+' | 프레임평균 '+r.mean.R0.toFixed(2)+' → '+r.mean[m].toFixed(2)+'px';im.src=r.image;im.alt=r.id;im.loading='lazy';a.append(h,im);root.append(a)}}for(const id of ['filter','model'])el(id).addEventListener('input',render);render();</script></html>'''.replace('__DATA__',payload)
    C.write_text(OUT/'index.html',page)
    C.freeze(H.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,
        decoder_tests_passed=7,prediction_preservation_checks=checks,CPU_saved_logit_redecode_max_abs_px=parity,
        recovered_over40= recoveries,all_over40_recoveries_same_C2_branch=True,
        full_gallery_frames=len(cards),new_labels=0,new_training=0,auto_promoted=False,
        evidence=[C.bound(__file__),C.bound(OUT/'index.html'),C.bound(OUT/'CASES.json'),C.bound(H.DOC/'RESULTS.json')]))
    print('HEATMAP_AUDIT_COMPLETE',checks,'parity',parity,'over40',recoveries,flush=True)
    print('GALLERY',OUT/'index.html',flush=True)


if __name__=='__main__':main()
