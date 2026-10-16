"""Audit and explain the point-neighborhood pilot without claiming trained fusion."""
from pathlib import Path
import hashlib
import html
import json
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from run import RUN, ROOT, MANIFEST, STOCK, V6, Split, sha, write


def read(p):return json.loads(p.read_text())


def main():
    torch.set_num_threads(2);data=Split(MANIFEST);summary=read(RUN/'SUMMARY.json')
    protocol=read(RUN/'PROTOCOL.json');assert sha(Path(__file__).parent/'run.py')==protocol['source_sha256']
    assert sha(STOCK)==protocol['stock_sha256'] and sha(MANIFEST)==protocol['manifest_sha256']
    stocks=read(STOCK)['rows'];boxes=[]
    for i in range(len(data.records)):
        O,_,_=data.batch([i],'cpu',targets=False);boxes.append(O['box'][0].numpy())
    analyses={};max_error=0.;count=0;preds={};metrics={}
    for seed in (1,2,3):
        assert sha(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt')==protocol['checkpoints'][str(seed)]
        p=read(RUN/f'predictions_seed{seed}.json');m=read(RUN/f'metrics_seed{seed}.json')
        assert p['gt_opened_during_inference']==0
        assert sha(RUN/f'predictions_seed{seed}.json')==m['predictions_sha256']
        preds[str(seed)]=p['rows'];metrics[str(seed)]=m['rows'];cells={}
        for name in summary['summaries'][str(seed)]:
            vals=[]
            for i,(pr,mr) in enumerate(zip(p['rows'],m['rows'][name])):
                assert pr['id']==mr['id']==data.records[i]['id']
                assert np.array_equal(np.array(pr['yolo_points']),np.asarray(stocks[i]['points'],dtype=np.float32))
                l=np.array(pr['cases'][name]['lines']);mass=np.array(pr['cases'][name]['mass'])
                normalized=l/np.linalg.norm(l[...,:2],axis=-1)[...,None]
                ep=np.array(mr['gt_endpoints']);valid=np.array(mr['valid_edges'])
                endpoints=np.concatenate((ep,np.ones((*ep.shape[:-1],1))),axis=-1)
                err=np.abs(np.einsum('epd,emd->epm',endpoints,normalized)).mean(1)
                top=mass.argmax(-1);topline=normalized[np.arange(12),top]
                cost=np.abs(np.einsum('epd,qd->epq',endpoints,topline)).mean(1)
                u,v=linear_sum_assignment(cost[valid]);role=err[np.arange(12),top]
                replay=dict(role_mean_px=role[valid].mean(),unordered_mean_px=cost[valid][u,v].mean(),
                    role_best3_mean_px=err.min(-1)[valid].mean(),horizontal_mean_px=role[:8][valid[:8]].mean(),
                    height_mean_px=role[8:][valid[8:]].mean())
                for k,z in replay.items():
                    max_error=max(max_error,abs(float(z)-mr[k]));assert abs(float(z)-mr[k])<1e-8
                points=np.concatenate((np.array(pr['yolo_points']),np.ones((8,1))),axis=-1)
                nearest=np.abs(points@topline.T).min(0);sigma=.03*np.linalg.norm(boxes[i][2:]-boxes[i][:2])
                bad_role=(role>10)&valid;bad_geom=cost[valid].min(0)>10
                vals.append(dict(id=pr['id'],pred_lines=12,gt_edges=int(valid.sum()),
                    near_points=int((nearest<=sigma).sum()),bad_role=int(bad_role.sum()),
                    bad_role_near=int((bad_role&(nearest<=sigma)).sum()),bad_geometry=int(bad_geom.sum()),
                    bad_geometry_near=int((bad_geom&(nearest<=sigma)).sum()),
                    covered_gt_edges=int((cost[valid].min(1)<=10).sum()),
                    prediction_matches_any_gt=int((cost[valid].min(0)<=10).sum())))
                count+=1
            total={k:sum(r[k] for r in vals) for k in vals[0] if k!='id'}
            total.update(prediction_near_point_fraction=total['near_points']/total['pred_lines'],
                wrong_role_near_fraction=total['bad_role_near']/max(1,total['bad_role']),
                wrong_geometry_near_fraction=total['bad_geometry_near']/max(1,total['bad_geometry']),
                gt_coverage10=total['covered_gt_edges']/total['gt_edges'],
                prediction_matches_any_gt_fraction=total['prediction_matches_any_gt']/total['pred_lines'])
            cells[name]=dict(totals=total,rows=vals)
        analyses[str(seed)]=cells
    write(RUN/'AUDIT.json',dict(status='PASS',replayed_frame_conditions=count,max_metric_delta=max_error,
        checkpoints_and_source_verified=True,point_coordinates_unchanged=True,prediction_count_per_seed=253,
        gt_inference_loads_per_seed=0,scope='Independent homogeneous-line arithmetic replay; not source GT certification.'))
    write(RUN/'NEIGHBORHOOD_ANALYSIS.json',dict(cells=analyses,scope='Post-hoc descriptive thresholds: '
        'line normal error>10px is wrong, proximity<=.03*box diagonal is near; no retuning from these diagnostics. '
        'Matching any GT is a relaxed diagnostic, not one-to-one detection or role accuracy.'))
    body=['# 점 주변 정보를 이용한 Deep Hough 후보 안내',
        '**점을 연결하지 않았다. YOLO 점 위치도 바꾸지 않았다. 점 근처를 지나는 선 후보에 부드러운 가중치를 주는 빠른 검사를 수행했지만, 현재 고정 DHT 점수에서는 안정적인 선 검출 개선이 없었다.**',
        '합성 calibration 253장, 기존 Hough 모델 3개, 비교 6조건. 학습된 attention의 효과나 실제 자세 개선을 평가한 결과는 아니다. 이번 검사에서 GT는 선 평가에만 사용했고, 점 prior는 원래 YOLO가 예측한 8개 좌표만 사용했다.',
        '## 실제로 검사한 방법',
        '각 직선 후보에서 가장 가까운 YOLO 점까지의 거리 d를 계산했다. 점 쌍이나 점 사이 선분은 만들지 않았다. 가중치는 0.1 + 0.9·exp(−d²/(2σ²))이며, 그 로그를 DHT 후보 점수에 더했다. σ는 관측 박스 대각선의 1%, 3%, 10%; 주 비교는 사전에 정한 3%다. 먼 선도 0으로 제거하지 않아 전체 선 증거를 남겼다.',
        '다른 이미지의 점을 현재 박스에 맞춰 옮긴 대조군과, 박스 코너·변 중앙 8개 지점을 사용한 대조군도 검사했다. 점 순서를 바꿔도 결과가 같고, 점 하나는 그곳을 지나는 여러 방향의 선을 똑같이 지원한다는 수치 검사를 통과했다.',
        '## 선 위치 오차 — 동일한 일대일 경계 매칭 평가(px)',
        '아래 지표는 꼭짓점 번호를 제외한 선의 공간 배치를 평가한다. GT의 각 선과 예측선 사이에서 양 끝점의 수직거리를 사용하고, 중복 예측을 구분하기 위해 동일한 일대일 매칭을 모든 조건에 적용했다. 점/자세 오차와는 다른 지표다.',
        '| 조건 | seed 1 | seed 2 | seed 3 |','|---|---:|---:|---:|']
    labels={'baseline':'기존 DHT','points_001':'YOLO 점 주변 1%','points_003':'YOLO 점 주변 3% — 주 비교',
            'points_010':'YOLO 점 주변 10%','donor_003':'다른 이미지 점 주변 3%','box_003':'박스 경계 지점 주변 3%'}
    for k,label in labels.items():body.append('| '+label+' | '+' | '.join(f'{summary["summaries"][str(s)][k]["unordered_mean_px"]["mean"]:.3f}' for s in (1,2,3))+' |')
    body += ['## 왜 이 단순한 방법으로는 부족했는가',
        '1. **기존 선 후보가 이미 점 근처를 지나는 경우가 많았다.** 역할이 틀린 선(정해진 경계에 대한 오차>10px) 중 점 반경 안을 지나는 비율은 아래와 같다. 점 주변이라는 조건만으로 이들을 구별하기 어렵다.',
        '| 기존 DHT 진단 | seed 1 | seed 2 | seed 3 |','|---|---:|---:|---:|']
    diaglabels={'prediction_near_point_fraction':'전체 선 중 점 주변을 지남',
        'wrong_role_near_fraction':'역할 오차가 큰 선 중 점 주변을 지남',
        'prediction_matches_any_gt_fraction':'어느 GT 경계든 하나에 가까운 예측선',
        'gt_coverage10':'예측선이 10px 안에서 포착한 GT 경계 비율'}
    for key,label in diaglabels.items():body.append('| '+label+' | '+' | '.join(f'{100*analyses[str(s)]["baseline"]["totals"][key]:.1f}%' for s in (1,2,3))+' |')
    body += ['2. **한 점의 주변이라는 정보는 선의 방향이나 역할을 결정하지 않는다.** 그 주변을 지나가는 여러 방향의 선과 주변 물체의 선도 점수를 받는다. 위치 힌트만 보강해도 어느 파렛트 경계인지 구분하는 문제는 남는다.',
        '3. **각 예측선이 어느 경계엔가 가깝다는 것과, 모든 경계를 서로 다르게 찾는 것은 다르다.** 여러 역할이 같은 경계 근처를 예측하면 다른 경계가 빠진다. 위의 느슨한 any-GT 일치율을 전체 선 검출 정확도로 해석하면 안 된다. 점 근접 조건 자체에는 역할 중복을 해소하는 정보가 없다.',
        '4. 반경을 1%로 좁히거나 10%로 넓혀도 세 모델의 일관된 개선은 없었다. 올바른 점을 쓴 주 비교가 다른 이미지 점 대조군보다도 매번 낫지는 않았다. 따라서 현재 결과로 점 위치 정보의 유효한 추가 이득을 입증하지 못했다.',
        '## 이 아이디어를 어디까지 판단할 수 있는가',
        '**확인할 가치가 있는 가설이지만, 지금 검사한 후보 가중치 방식은 통과하지 못했다.** 이 결과가 점 위치를 입력받아 영상 특징·선 방향을 함께 학습하는 아키텍처까지 부정하는 것은 아니다. 이번 방법은 기존 DHT logits에 위치 prior를 더했으며 특징 추출이나 voting 자체를 재학습하지 않았다.',
        '학습형 후속 구조를 만든다면 점 주변의 영상 특징으로 선의 방향·역할을 판단하고, 전체 선 구간의 증거도 유지해야 한다. 이런 학습형 구조의 개선 여부는 이번 검사로 확인하지 않았으며 성공을 약속하지 않는다. 현재 원래 YOLO와 생산 추론은 그대로 보존했다.',
        f'수치 감사 {count:,}개 프레임·조건 PASS, 최대 지표 재계산 차이 {max_error:.2e}px. 원래 점 좌표 보존과 추론 중 GT 접근 0건을 확인했다. 기존 매칭 성공 253/256장에 대한 재사용 calibration 진단이며 새 holdout/실사/최종 자세 성능은 평가하지 않았다. 투영 길이<1px인 경계는 모든 조건에서 같은 기준으로 제외했다.',
        '전체 수치: SUMMARY.json. 원인 진단: NEIGHBORHOOD_ANALYSIS.json. 수치 감사: AUDIT.json. 원본 예측 및 프레임별 점수도 동일 폴더에 보존했다.']
    text=body[0]
    for previous,block in zip(body,body[1:]):text+=('\n' if previous.startswith('|') and block.startswith('|') else '\n\n')+block
    (RUN/'REPORT_KO.md').write_text(text+'\n')
    # Visualize actual points, neighborhoods, and infinite-line predictions. No point-connecting lines.
    metrics1=metrics['1'];delta={r['id']:s['unordered_mean_px']-r['unordered_mean_px'] for r,s in zip(metrics1['baseline'],metrics1['points_003'])}
    fixed=sorted(delta,key=lambda k:hashlib.sha256(('neighborhood-v8-gallery:'+k).encode()).digest())[:2]
    selected=list(dict.fromkeys(fixed+[min(delta,key=delta.get),max(delta,key=delta.get)]));gallery=[]
    (RUN/'figures').mkdir(exist_ok=True);indices={r['id']:i for i,r in enumerate(data.records)}
    for fid in selected:
        i=indices[fid];rec=data.records[i];pr=preds['1'][i];bb=boxes[i];sigma=.03*np.linalg.norm(bb[2:]-bb[:2])
        img=np.asarray(Image.open(ROOT/rec['original_image']).convert('RGB'))[100:-100,100:-100]
        pts=np.array(pr['yolo_points']);ep=np.array(metrics1['baseline'][i]['gt_endpoints'])
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for ax in axes:
            ax.imshow(img);ax.set_xlim(0,img.shape[1]);ax.set_ylim(img.shape[0],0);ax.axis('off')
            ax.scatter(pts[:,0],pts[:,1],c='#ffdc50',s=16,zorder=6)
        for x,y in pts:axes[0].add_patch(Circle((x,y),sigma,facecolor='#ff70bc',edgecolor='#ff70bc',alpha=.28))
        axes[0].set_title('YOLO points + local neighborhoods\nNo point pairs or connecting segments',fontsize=10)
        for ax,name in zip(axes[1:],('baseline','points_003')):
            for seg in ep:ax.plot(seg[:,0],seg[:,1],color='#5fe5ff',lw=1.3)
            l=np.array(pr['cases'][name]['lines']);mass=np.array(pr['cases'][name]['mass']);top=l[np.arange(12),mass.argmax(-1)]
            # Clip infinite candidates to observed ROI; endpoints only represent display clipping.
            for a,b,c in top:
                cut=[]
                if abs(b)>1e-10:
                    for x in [bb[0],bb[2]]:
                        y=-(a*x+c)/b
                        if bb[1]-1e-6<=y<=bb[3]+1e-6:cut.append((x,y))
                if abs(a)>1e-10:
                    for y in [bb[1],bb[3]]:
                        x=-(b*y+c)/a
                        if bb[0]-1e-6<=x<=bb[2]+1e-6:cut.append((x,y))
                if len(cut)>=2:
                    q=np.array(cut);pair=max(((u,v) for u in range(len(q)) for v in range(u+1,len(q))),key=lambda t:np.linalg.norm(q[t[0]]-q[t[1]]))
                    q=q[list(pair)];ax.plot(q[:,0],q[:,1],color='#ff8650',lw=1,alpha=.8)
            ax.set_title(('Original DHT' if name=='baseline' else 'Point-neighborhood guided DHT')+f'\nmatched line error {metrics1[name][i]["unordered_mean_px"]:.2f}px',fontsize=10)
        reason='fixed SHA sample' if fid in fixed else ('largest gain' if delta[fid]<0 else 'largest harm')
        fig.suptitle(f'{fid} | {reason} | yellow: unchanged YOLO points; cyan: GT edges; orange: predicted lines',fontsize=10)
        fig.tight_layout();path=RUN/'figures'/f'{fid}.png';fig.savefig(path,dpi=120);plt.close(fig)
        gallery.append(dict(id=fid,path=str(path.relative_to(RUN)),selection=reason,delta_px=delta[fid]))
    write(RUN/'GALLERY.json',gallery)
    table='<table><tr><th>선 위치 오차(px)</th><th>seed 1</th><th>seed 2</th><th>seed 3</th></tr>'
    for k,label in labels.items():table+='<tr><td>'+label+'</td>'+''.join(f'<td>{summary["summaries"][str(s)][k]["unordered_mean_px"]["mean"]:.3f}</td>' for s in (1,2,3))+'</tr>'
    table+='</table>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>점 주변으로 선 탐색 안내 — 검증</title><style>body{font:16px/1.65 system-ui;max-width:1500px;margin:30px auto;padding:0 24px;background:#111827;color:#edf2fa}a{color:#88caff}td,th{border:1px solid #53637a;padding:9px;text-align:left}table{border-collapse:collapse;width:100%}img{width:100%}pre{white-space:pre-wrap;font:15px/1.7 system-ui}select{padding:9px;font:inherit}.note{background:#273449;border-left:4px solid #ffc464;padding:15px}</style><h1>점 주변에 선이 있다는 정보를 사용할 수 있는가?</h1><p class="note">점 연결 없이, 기존 YOLO 점 주변을 지나는 DHT 후보에 가중치를 주었습니다. 고정 모델 3개에서 일관된 개선은 없었습니다. 학습형 attention의 성능을 검증한 결과는 아닙니다.</p>'''+table
    page+='<h2>실제 점 주변과 선 후보</h2><select id="choose">'+''.join(f'<option value="{g["path"]}">{g["id"]} | {g["selection"]}</option>' for g in gallery)+'</select>'
    page+='<p><img id="panel" src="'+gallery[0]['path']+'" alt="점 주변 영역, 원래 DHT, 안내 적용 DHT"></p><p><a href="REPORT_KO.md">전체 보고서</a> · <a href="SUMMARY.json">수치</a> · <a href="AUDIT.json">독립 검산</a></p><pre>'+html.escape(text)+'</pre><script>document.getElementById("choose").addEventListener("change",e=>document.getElementById("panel").src=e.target.value)</script></html>'
    (RUN/'index.html').write_text(page)
    print('AUDIT',count,max_error)
    print(json.dumps({s:c['baseline']['totals'] for s,c in analyses.items()},indent=2))


if __name__=='__main__':main()
