"""Independent NumPy metric replay and reviewable causal diagnostic report."""
from pathlib import Path
import hashlib
import html
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import linear_sum_assignment
from run import RUN, ROOT, V6, MANIFEST, OLD, Split, sha, write


def read(p):return json.loads(p.read_text())


def audit(docs, weight):
    data=Split(MANIFEST);records={r['id']:i for i,r in enumerate(data.records)}
    targets={};max_norm=0.;max_raw=0.;max_geom=0.;max_projection=0.;count=0
    signs=np.array([[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1]])
    perm=[2,3,0,1,6,7,4,5]
    for fid,i in records.items():
        O,S,order=data.batch([i],'cpu')
        targets[fid]=(O,S,int(order[0]))
    for group in [d['cells'] for d in docs.values()]+[d['cells'] for d in weight['cells'].values()]:
        for key,rows in group.items():
            assert [r['id'] for r in rows]==list(records), key
            for row in rows:
                O,S,order=targets[row['id']];uv=np.array(row['projected_points']);gt=S['points'][0,:8].numpy().astype(float)
                if 'R' in row:
                    d=O['dims'][0].numpy().astype(float);X=signs*d[[0,2,1]]*.5
                    cam=X@np.array(row['R']).T+np.array(row['t']);p=cam@O['K'][0].numpy().T
                    reproj=p[:,:2]/np.maximum(cam[:,2:],1e-6)
                    max_projection=max(max_projection,float(abs(reproj-uv).max()))
                    assert np.allclose(reproj,uv,atol=.003,rtol=2e-5),row['id']
                errs=[np.linalg.norm(uv-gt,axis=-1)]
                if order==2:errs.append(np.linalg.norm(uv[perm]-gt,axis=-1))
                diag=np.linalg.norm(O['image_hw'][0].numpy().astype(float))
                norms=[np.minimum(e/diag,1.).mean() for e in errs];best=int(np.argmin(norms))
                norm=norms[best] if row['valid'] else 1.
                max_norm=max(max_norm,abs(norm-row['normalized']))
                assert abs(norm-row['normalized'])<2e-6
                if row['valid']:
                    max_raw=max(max_raw,abs(errs[best].mean()-row['raw_px']))
                    cost=np.linalg.norm(uv[:,None]-gt[None],axis=-1);u,v=linear_sum_assignment(cost)
                    max_geom=max(max_geom,abs(cost[u,v].mean()-row['unordered_px']))
                    assert abs(errs[best].mean()-row['raw_px'])<.002
                    assert abs(cost[u,v].mean()-row['unordered_px'])<.002
                count+=1
    lock=read(V6/'PROTOCOL_LOCK.json')
    for name,digest in lock['source_sha256'].items():assert sha(OLD/'plpose_local'/name)==digest
    protocol=read(RUN/'PROTOCOL.json')
    assert sha(Path(__file__).parent/'run.py')==protocol['source_sha256']
    for s,digest in protocol['checkpoints'].items():assert sha(V6/f'cache_heads/hough_seed{s}/checkpoint_final.pt')==digest
    result=dict(status='PASS',replayed_frame_cases=count,max_normalized_delta=max_norm,max_raw_px_delta=max_raw,
                max_unordered_px_delta=max_geom,max_projection_delta=max_projection,
                full_id_coverage=True,frozen_v6_geometry_unchanged=True,checkpoint_bindings=True,
                caveat='Numerical consistency audit; does not independently certify source GT or new generalization.')
    write(RUN/'AUDIT.json',result)
    return result


def paired(docs, a, b):
    delta=np.array([[r['raw_px']-q['raw_px'] for r,q in zip(d['cells'][b],d['cells'][a])] for d in docs.values()])
    x=delta.mean(0);rng=np.random.default_rng(709)
    boots=x[rng.integers(0,len(x),(3000,len(x)))].mean(1)
    return dict(mean_change_px=float(x.mean()),frame_bootstrap_ci95=np.quantile(boots,[.025,.975]).tolist(),
                per_seed_mean_change_px=delta.mean(1).tolist(),per_seed_worsened_frames=(delta>0).sum(1).tolist(),
                scope='Seeds averaged within frame before resampling; descriptive reused-calibration interval.')


def gallery(docs):
    d=docs['1'];data=Split(MANIFEST);idx={r['id']:i for i,r in enumerate(data.records)}
    base=d['cells']['stock_oracle_order/pnp/none/5'];line=d['cells']['stock_oracle_order/pnp/predicted/5']
    delta={r['id']:s['raw_px']-r['raw_px'] for r,s in zip(base,line)}
    fixed=sorted(idx,key=lambda s:hashlib.sha256(('causal-v7:'+s).encode()).digest())[:2]
    chosen=list(dict.fromkeys(fixed+[max(delta,key=delta.get),min(delta,key=delta.get)]))
    folder=RUN/'figures';folder.mkdir(exist_ok=True);out=[]
    keys=['stock_oracle_order/pnp/none/5','stock_oracle_order/pnp/predicted/5','stock_oracle_order/pnp/oracle/5']
    labels=['YOLO points -> pose','Same points + predicted Hough lines','Same points + exact GT lines (oracle)']
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for fid in chosen:
        i=idx[fid];rec=data.records[i];O,S,_=data.batch([i],'cpu')
        img=np.asarray(Image.open(ROOT/rec['original_image']).convert('RGB'))[100:-100,100:-100]
        gt=S['points'][0,:8].numpy();fig,axes=plt.subplots(1,4,figsize=(19,4.5))
        for ax in axes:ax.imshow(img);ax.axis('off');ax.set_xlim(0,img.shape[1]);ax.set_ylim(img.shape[0],0)
        axes[0].set_title('Original synthetic image')
        for ax,key,label in zip(axes[1:],keys,labels):
            row=d['cells'][key][i];uv=np.array(row['projected_points'])
            for u,v in edges:
                ax.plot(gt[[u,v],0],gt[[u,v],1],color='#50e7ff',lw=1.2)
                ax.plot(uv[[u,v],0],uv[[u,v],1],color='#ff9b35',lw=1.5)
            ax.set_title(label+f'\nrole error {row["raw_px"]:.2f}px',fontsize=10)
        reason='fixed SHA sample' if fid in fixed else ('largest harm' if delta[fid]>0 else 'largest gain')
        fig.suptitle(f'{fid} | {reason} | ALL pose panels use oracle corner correspondence; cyan GT, orange output',fontsize=10)
        fig.tight_layout();path=folder/(fid+'.png');fig.savefig(path,dpi=120);plt.close(fig)
        out.append(dict(id=fid,path=str(path.relative_to(RUN)),selection=reason,delta_px=delta[fid]))
    return out


def main():
    torch.set_num_threads(2)
    docs={str(s):read(RUN/f'seed{s}.json') for s in (1,2,3)};weight=read(RUN/'WEIGHT_PROBE.json')
    checked=audit(docs,weight)
    comparisons={
        'strong_points_predicted_lines':paired(docs,'stock_oracle_order/pnp/none/5','stock_oracle_order/pnp/predicted/5'),
        'strong_points_oracle_lines':paired(docs,'stock_oracle_order/pnp/none/5','stock_oracle_order/pnp/oracle/5'),
        'point_derived_start':paired(docs,'new/learned/predicted/5','new/pnp/predicted/5'),
        'extra_iterations':paired(docs,'new/learned/predicted/5','new/learned/predicted/25'),
    }
    weights={str(s):{k:v['raw_mean'] for k,v in weight['cells'][str(s)]['summary'].items()} for s in (1,2,3)}
    result=dict(status='CURRENT_PREDICTED_LINES_HARM_STRONG_POINTS',comparisons=comparisons,weight_results=weights,
                scope='Controlled diagnostic, not deployable fusion: strong-point IDs use oracle assignment. '
                      'No training, no real images, no new held-out evaluation; original YOLO remains unchanged.',
                audit_sha256=sha(RUN/'AUDIT.json'))
    write(RUN/'CONCLUSION.json',result)
    body=['# 점·선 융합 실패 원인 추가 검증',
          '**현재 예측선을 그대로 결합하면 좋은 YOLO 점도 악화된다. 정확한 선을 넣으면 개선되므로, 결합 자체의 불가능이 아니라 관측 정확도와 잘못된 선의 영향이 핵심 문제다.**',
          '합성 calibration 253장 × 고정 Hough 모델 3개 × 18개 교체 조건을 검사했다. 추가로 선 가중치 5개와 반복 25회 조건을 검사했다. 학습·원본 모델·수작업 GT는 변경하지 않았다.',
          '**아래 YOLO 점 조건은 GT로 꼭짓점 번호를 맞춘 특권 진단이다. 실제 적용 가능한 아키텍처의 성능이 아니다.** 이전 val505장의 4.18px와는 평가 집합·지표가 다르므로 직접 비교하지 않는다. 이번 원본 점의 기하 오차 평균은 '+f'{np.mean([r["stock_oracle_order_unordered_px"] for r in docs["1"]["observations"]]):.3f}px다.',
          '## 원인 분리 결과 — 프레임별 코너 평균오차(px)',
          '| 조건 | seed 1 | seed 2 | seed 3 |','|---|---:|---:|---:|']
    names={
        'new/learned/none/5':'새 점 + 학습한 초기 자세, 선 없음',
        'new/learned/predicted/5':'새 점 + 학습한 초기 자세 + 예측선',
        'new/learned/predicted/25':'동일 조건, 반복만 5→25회',
        'new/pnp/predicted/5':'새 점 유지, 초기 자세만 점 기반 PnP로 변경',
        'gt/learned/none/5':'정답 점만 교체 + 학습한 초기 자세, 선 없음',
        'gt/pnp/none/5':'정답 점 + PnP, 선 없음 (수치 대조군)',
        'gt/pnp/predicted/5':'정답 점 + PnP + 예측선',
        'stock_oracle_order/pnp/none/5':'YOLO 점 + PnP, 선 없음 (번호는 GT 사용)',
        'stock_oracle_order/pnp/predicted/5':'동일 YOLO 점·초기 자세 + 예측선',
        'stock_oracle_order/pnp/gt_best_mode/5':'동일 조건 + GT가 고른 기존 선 후보',
        'stock_oracle_order/pnp/oracle/5':'동일 조건 + 정확한 정답선',
    }
    for k,label in names.items():body.append('| '+label+' | '+' | '.join(f'{d["summary"][k]["raw_mean"]:.3f}' for d in docs.values())+' |')
    body += ['## 확인한 사실과 아직 확정하지 못한 이유',
        '1. **새 점 관측 오차는 실제 병목이다.** 학습 초기 자세를 고정하고 점만 정답으로 교체하면, 선 없는 오차 94~99px가 0.95~1.94px로 줄었다. 초기 자세가 항상 무용한 것은 아니다. 새 head의 표현·학습량·손실 중 단일 원인은 이 검사만으로 확정하지 못한다.',
        '2. **부정확한 선이 좋은 점을 끌어가는 현상은 직접 확인했다.** 정답 점도 예측선을 넣으면 10.7~12.2px 오차가 생긴다. GT로 번호를 맞춘 실제 YOLO 점에서도 약 2.96px가 11.66~13.01px로 악화한다. 이는 점·초기 자세가 약해서만 생기는 문제가 아니다.',
        '3. **정확한 선은 좋은 점에도 추가 정보를 줄 수 있다.** 같은 YOLO 점·초기 자세에서 정답선은 약 2.96→1.98px로 개선한다. 점과 선이 정보를 공유한다는 이유만으로 결합이 불가능한 것은 아니다. 단, GT선은 실제로 얻을 수 있는 예측 성능이 아니다.',
        '4. **초기 자세 계산 교체·반복 횟수 증가만으로는 해결되지 않았다.** 실제 예측점을 사용한 PnP 교체는 세 seed 모두 예측선 결합 결과를 악화시켰다. 반복 5→25회도 일관된 개선이 없었다. 다른 초기화 방법 전체를 배제한 것은 아니다.',
        '5. **좋은 후보 선택만으로도 부족했다.** GT로 각 역할의 가장 가까운 기존 선을 골라도 정확한 YOLO 점 조건에서 10.68~11.16px였다. 기존 후보 자체의 오차, 역할 대응, 여러 선의 공동 기하 일관성이 남는다. 어느 한 항목의 원인 비율을 분리한 것은 아니다.',
        '6. **현재 결합은 역할별 선의 신뢰도를 보장하지 않는다.** 코드의 mass는 남겨진 후보 사이 상대 비중이며, 그 경계가 실제로 맞다는 확률이 아니다. 모든 역할이 기본적으로 활성화되고 점·선의 절대 신뢰도 보정은 없다. 이는 관측된 악화와 맞는 설명이다. Gradient surgery는 이번 추론 단계의 잘못된 관측을 직접 고치지 못한다.',
        '7. 가려진 경계와 구조선의 라벨 불일치, 실사 도메인 차이는 이번 추가 검사에서도 직접 검증하지 않았다. 합성 정답 점 조건에서도 실패하므로 사용자의 실제 수작업 annotation 오차가 이번 현상의 필수 원인은 아니다.',
        '## 선 영향력만 줄이는 추가 검사',
        '위 18조건 결과를 본 후 수행한 탐색적 후속 검사다. 같은 GT 번호 정렬 YOLO 점·PnP 초기 자세·예측선을 유지하고 역할 전체의 선 가중치만 변경했다. 이 자료로 가중치를 선택하거나 실사 개선을 주장하지 않는다.',
        '| 선 가중치 / 반복 | seed 1 | seed 2 | seed 3 |','|---|---:|---:|---:|']
    for k in weight['cells']['1']['summary']:body.append('| '+k+' | '+' | '.join(f'{weights[str(s)][k]:.5f}' for s in (1,2,3))+' |')
    body += ['가중치를 0.1로 줄여도 2.956px→3.27~3.31px로 악화했다. 0.01·0.001에서는 선을 거의 무시하는 결과로 돌아왔고 세 seed 모두의 안정적인 개선은 없었다. 따라서 이번 고정 전역 가중치 조정만으로 해결됐다고 판단하지 않는다.',
        '가중치 검사는 CPU에서 각 배치의 관측을 한 번 계산해 모든 가중치에 공유했다. 앞의 18조건 검사는 GPU에서 실행했다. 동일 가중치 1의 두 실행 간 평균 차이는 seed3에서 약 0.26px였으므로 비트 동일 재현이나 미세한 수치 차이의 유의성을 주장하지 않는다. 가중치의 효과는 반드시 같은 CPU 실행 내에서 비교한다. 두 환경 모두 큰 악화라는 결론은 동일하다.',
        '## 검증 범위와 개발 판단',
        f'독립 NumPy 재계산 {checked["replayed_frame_cases"]:,}개 프레임·조건 통과. 정답 점 대조군 P90<0.01px, 모든 PnP 입력에서 계산 실패 0건이다. 이전 v6 수치 코어와 모델 SHA를 확인했다. GPU/CPU 원래 대조군 평균 차이는 최대 약 0.01px로 기록했으며 비트 일치를 주장하지 않는다.',
        '253장은 기존 검출 매칭이 된 calibration 집합이다(요청 256장 중 제외 3장). 동일 프레임 내 원인 비교용이며 전체 검출 성능을 나타내지 않는다. 대칭은 기존 C1/C2 계약을 유지했다. GT 내부 투영 일치가 실제 영상 경계 annotation의 정확성을 보장하지 않는다.',
        '**현재 구조를 그대로 확대 학습하거나 YOLO에 예측선을 무조건 추가하는 것은 지지되지 않는다.** 기존 YOLO 점 head 보존은 필요하지만 그것만으로 해결되지는 않는다. 다음 구조는 올바른 경계 역할·선 정확도와 점/선의 상대 신뢰도를 학습해 정상점을 지키는 조건부터 통과해야 한다. 이러한 구조의 성공은 아직 입증하지 않았다.',
        '기존 불확실성 가중 결합 실험에서 반복 사용한 실사 DEV의 제한적인 개선 사례는 있다. 따라서 모든 가중 결합이 실패했다고 일반화하지 않는다. 이번 공동 자세 구조에서 새 end-to-end 개선을 입증한 결과는 아니다. 관련 기록: `_docs/notes/dht_uncertainty_fusion.md`.',
        'PnP 구현 근거: [OpenCV 공식 문서](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html). 이번 SQPnP는 입력 2D점·3D코너·카메라만 사용하며 RANSAC 필터나 self-training을 실행하지 않는다.',
        '모든 수치: SUMMARY.json, WEIGHT_PROBE.json. 프레임별 결과: seed1.json~seed3.json. 수치 감사: AUDIT.json.']
    text=body[0]
    for previous,block in zip(body,body[1:]):
        text+=('\n' if previous.startswith('|') and block.startswith('|') else '\n\n')+block
    text+='\n';(RUN/'REPORT_KO.md').write_text(text)
    panels=gallery(docs)
    # Native HTML table for readable side-by-side metrics, original full report below.
    table='<table><tr><th>조건</th><th>seed 1</th><th>seed 2</th><th>seed 3</th></tr>'
    for k,label in names.items():table+='<tr><td>'+html.escape(label)+'</td>'+''.join(f'<td>{d["summary"][k]["raw_mean"]:.3f}px</td>' for d in docs.values())+'</tr>'
    table+='</table>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>점·선 실패 원인 추가 검증</title>
<style>body{font:16px/1.65 system-ui;margin:32px auto;max-width:1450px;padding:0 24px;background:#101621;color:#e9edf5}h1{font-size:29px}a{color:#82c6ff}table{border-collapse:collapse;width:100%;background:#1b2636}td,th{padding:10px;border:1px solid #46566d;text-align:left}img{width:100%;background:white}pre{white-space:pre-wrap;font:15px/1.7 system-ui}.note{border-left:4px solid #ffc763;padding:15px;background:#273344}select{font:inherit;padding:8px;max-width:100%}</style>
<h1>점·선 융합: 실패 원인을 분리해서 검증</h1><p class="note">합성 253장, 3개 고정 모델. 정확한 점도 잘못된 예측선 때문에 악화했습니다. 아래 YOLO 점 조건은 GT로 번호를 정렬한 원인 진단이며, 실제 적용 가능한 모델 성능이 아닙니다.</p>'''+table
    page+='<h2>같은 점에서 선만 바꾼 사례</h2><p>청록: GT, 주황: 결과. 모든 자세 패널은 GT 꼭짓점 대응을 사용한 특권 진단입니다.</p><select id="pick">'
    for p in panels:page+=f'<option value="{html.escape(p["path"])}">{html.escape(p["id"]+" | "+p["selection"])}</option>'
    page+='</select><p><img id="panel" alt="원본, 선 없음, 예측선, 정답선 비교" src="'+panels[0]['path']+'"></p>'
    page+='<p><a href="REPORT_KO.md">전체 결과</a> · <a href="CONCLUSION.json">수치 요약</a> · <a href="AUDIT.json">독립 검산</a></p><pre>'+html.escape(text)+'</pre>'
    page+='<script>document.getElementById("pick").addEventListener("change",e=>document.getElementById("panel").src=e.target.value);</script></html>'
    (RUN/'index.html').write_text(page);write(RUN/'GALLERY.json',panels)
    print(json.dumps(result,ensure_ascii=False,indent=2));print('AUDIT',checked)


if __name__=='__main__':main()
