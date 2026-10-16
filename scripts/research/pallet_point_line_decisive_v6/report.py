"""Source-backed decision report and explicitly labelled diagnostic figures."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import sys
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
sys.path.insert(0,str(KIT))
from diagnostics import load_model, Split, G, choose_oracle_symmetry
from stock_geometry_baseline import set_error


def read(p):return json.loads(Path(p).read_text())


def aggregate(run):
    verdict=read(run/'VERDICT.json');diagnosis=read(run/'DIAGNOSTICS.json')
    rows=[]
    for arm in ('point','direct','hough'):
        for seed in (1,2,3):
            d=read(run/'evaluations'/f'{arm}_seed{seed}_synth_val.json')
            rows.append(dict(arm=arm,seed=seed,primary=d['primary_normalized_mean'],
                frames=d['frames'],coverage=d['coverage'],**d['conditional_raw_px']))
    return verdict,diagnosis,rows


def make_gallery(run):
    data=Split(ROOT/'data/pallet/results/pallet_point_line_v5_local/export/synth_val.json')
    models={};pred={};ev={}
    for arm in ('point','hough'):
        models[arm]=load_model(run/'cache_heads'/f'{arm}_seed1/checkpoint_final.pt','cpu')[0]
        pred[arm]={r['id']:r for r in read(run/'predictions'/f'{arm}_seed1_synth_val.json')['rows']}
        ev[arm]={r['id']:r for r in read(run/'evaluations'/f'{arm}_seed1_synth_val.json')['per_frame']}
    ids=[r['id'] for r in data.records]
    fixed=sorted(ids,key=lambda s:hashlib.sha256(('decisive-v6-gallery:'+s).encode()).digest())[:4]
    delta={fid:ev['hough'][fid]['normalized']-ev['point'][fid]['normalized'] for fid in ids}
    chosen=list(dict.fromkeys(fixed+sorted(ids,key=delta.get)[:2]+sorted(ids,key=delta.get)[-2:]))
    gallery=[];folder=run/'figures';folder.mkdir(exist_ok=True)
    with torch.inference_mode():
        for fid in chosen:
            i=ids.index(fid);rec=data.records[i]
            O,_,orders=data.batch([i],'cpu',targets=False)
            out=models['hough'](O)
            _,sup,_=data.batch([i],'cpu',targets=True)
            image=np.asarray(Image.open(ROOT/rec['original_image']).convert('RGB'))
            h,w=map(int,O['image_hw'][0])
            assert image.shape[:2]==(h+200,w+200)
            image=image[100:-100,100:-100]
            fig,axes=plt.subplots(1,4,figsize=(19,4.6))
            for ax in axes:
                ax.imshow(image);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
            axes[0].set_title('Original synthetic image')
            for ax,arm,color in ((axes[1],'point','#ffad32'),(axes[2],'hough','#54ff77')):
                p=pred[arm][fid];R=torch.tensor(p['R'])[None];t=torch.tensor(p['t'])[None]
                best=None
                for g in range(int(orders[0])):
                    uv=G.project(O['K'],R@G.yaw(2*g).T,t,G.corners(O['dims']))[0][0,:8]
                    err=float(torch.linalg.vector_norm(uv-sup['points'][0,:8],dim=-1).mean())
                    if best is None or err<best[0]:best=(err,uv.numpy())
                gt=sup['points'][0,:8].numpy()
                for u,v in G.EDGES:
                    ax.plot(gt[[u,v],0],gt[[u,v],1],color='#4fe7ff',lw=1,alpha=.8)
                    ax.plot(best[1][[u,v],0],best[1][[u,v],1],color=color,lw=1.5)
                ax.scatter(best[1][:,0],best[1][:,1],s=14,c=color)
                for k in range(8):
                    ax.annotate(str(k),gt[k],xytext=(3,-5),textcoords='offset points',color='#4fe7ff',fontsize=8,
                        bbox=dict(facecolor='black',alpha=.35,edgecolor='none',pad=.2))
                    ax.annotate(str(k),best[1][k],xytext=(3,4),textcoords='offset points',color=color,fontsize=8,
                        bbox=dict(facecolor='black',alpha=.35,edgecolor='none',pad=.2))
                seterr,_=set_error(best[1],gt)
                ax.set_title(f'{arm}: role error {best[0]:.1f}px\nunordered geometry {seterr.mean():.1f}px')
            lines=out['lines'][0];top=out['mass'][0].argmax(-1)
            pp=out['points'][0]
            for edge,(u,v) in enumerate(G.EDGES):
                l=lines[edge,top[edge]];n=l[:2];norm=(n*n).sum().clamp_min(1e-9)
                a=pp[u]-(pp[u]@n+l[2])/norm*n
                b=pp[v]-(pp[v]@n+l[2])/norm*n
                axes[3].plot([float(a[0]),float(b[0])],[float(a[1]),float(b[1])],lw=1.3,alpha=.8,color='#ff69ed')
            axes[3].set_title('Hough: highest-mass line per role')
            reason='fixed SHA sample' if fid in fixed else ('largest gain' if delta[fid]<0 else 'largest loss')
            fig.suptitle(f'{fid} | {reason} | cyan: GT; C2-aligned pose for evaluation only',fontsize=11)
            fig.tight_layout()
            p=folder/(fid.replace('/','_')+'.png');fig.savefig(p,dpi=130);plt.close(fig)
            gallery.append(dict(id=fid,path=str(p.relative_to(run)),selection=reason,delta=delta[fid]))
    return gallery


def main(args):
    torch.set_num_threads(2)
    run=Path(args.run)
    verdict,diag,rows=aggregate(run)
    comparisons={k:read(run/f'{k}.json') for k in ('H_VS_P','H_VS_D','D_VS_P')}
    hpass=comparisons['H_VS_P']['numeric_gate_passed'];dpass=comparisons['D_VS_P']['numeric_gate_passed']
    status='GO_TO_REAL_REVIEW' if hpass else ('DIRECT_LINE_ONLY_SIGNAL' if dpass else 'NO_GO_WITHIN_REGISTERED_BUDGET')
    gallery=make_gallery(run)
    stock_backstop=None
    labels={'point':'점 전용','direct':'점+직접 선','hough':'점+Hough'}
    lines=['# 점·선 융합 고정 비교 결과','',f'판정: **{status}**. 실사 개선 여부는 이번 합성 결과만으로 확정하지 않는다.','',
        '동일 frozen backbone·치수·C1/C2 계약에서 3개 구조×3개 seed, 각 2,000 update를 비교했다. '
        '전체 YOLO 공동학습이 아니라 새로운 공동 자세 head의 제한 학습이다.','',
        '합성 평가 512장 중 캐시 505장과 검출/매칭 실패 7장을 모두 분모에 포함한다. '
        'raw px 통계는 유효 자세가 있는 프레임별 8코너 평균 오차다. 기존 전체 점 P90=41.49px와 지표·모델이 달라 직접 비교할 수 없다.','',
        '| 구조 | seed | 정규화 오차↓ | raw 평균 px↓ | raw 중앙값 px↓ | raw P90 px↓ | coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {labels[r['arm']]} | {r['seed']} | {r['primary']:.6f} | {r['mean']:.2f} | {r['median']:.2f} | {r['p90']:.2f} | {r['coverage']:.3%} |")
    lines+=['','판정 기준은 모든 seed에서 평균 1% 개선, median/P90/coverage 비악화, '
        '좋았던 프레임(≤10px)을 >50px/pose 실패로 만드는 새 사례 0이다. 기준은 결과 전에 고정했다.','',
        '| 비교 | 모든 seed gate 통과 |','|---|---|']
    for name,c in comparisons.items():lines.append(f"| {name} | {c['numeric_gate_passed']} |")
    binding=read(run/'BINDING_CHECK.json')
    lines+=['',f"학습 파라미터: 점 {binding['trainable']['point']:,}, 직접 선 {binding['trainable']['direct']:,}, "
            f"Hough {binding['trainable']['hough']:,}. 공통 초기화·샘플 순서는 seed마다 같지만 "
            '총 파라미터 수까지 일치한 비교는 아니다. 따라서 직접 선과 Hough 차이를 순수 연산자 하나의 인과 효과로 단정하지 않는다.']
    if (run/'POINT_GEOMETRY_FIT_DIAGNOSTIC.json').exists():
        fit=read(run/'POINT_GEOMETRY_FIT_DIAGNOSTIC.json')
        lines+=['','## 새 점 head의 학습 적합도','',
                '각 split에서 SHA로 고정한 128장을 검사했다. 번호를 평가에서 자유롭게 '
                '맞춰도 학습 이미지에 위치 오차가 남는지 확인한다.','',
                '| seed | split | 역할 포함 자세 오차 px | 번호 제외 자세 배치 px | 번호 제외 점 관측 px |',
                '|---|---|---:|---:|---:|']
        for seed,c in fit['cells'].items():
            for split,r in c.items():
                m=r['means']
                lines.append(f"| {seed} | {split} | {m['role_aware_raw_px']:.2f} | "
                    f"{m['unordered_pose_raw_px']:.2f} | {m['unordered_point_observation_raw_px']:.2f} |")
        lines+=['','학습 split에서도 큰 오차가 남으면 합성→실사 차이만으로 실패를 설명할 수 없다. '
                '이는 충분한 학습/표현/번호 대응을 확보하지 못했다는 증거이며, 추가 학습만으로 '
                '반드시 해결된다는 뜻은 아니다. 이 약한 기준을 이긴 것만으로 원래 YOLO 개선을 선언하지 않는다.']
    if (run/'QUANTIZATION_DIAGNOSTIC.json').exists():
        q=read(run/'QUANTIZATION_DIAGNOSTIC.json')
        lines+=['','정답 선을 현재 학습 격자(36×65)의 bin 중심에 놓은 비학습 진단에서는 '
                f"normal 오차 중앙값 {q['results']['36x65']['median_px']:.2f}px, "
                f"P90 {q['results']['36x65']['p90_px']:.2f}px였다. "
                '따라서 수십~수백 픽셀 실패를 격자 해상도만으로 설명할 수 없다. '
                'neural decoder는 sub-bin 보간이 가능하므로 이 값은 정확도의 수학적 하한도 아니다.']
    if (run/'STOCK_GEOMETRY_BASELINE.json').exists():
        baseline=read(run/'STOCK_GEOMETRY_BASELINE.json')
        data=Split(ROOT/'data/pallet/results/pallet_point_line_v5_local/export/synth_val.json')
        truth={};observations={}
        for i,r in enumerate(data.records):
            O,T,_=data.batch([i],'cpu',targets=True)
            observations[r['id']]=O;truth[r['id']]=T['points'][0,:8].numpy()
        geom={'stock':dict(mean=baseline['raw_frame_mean'],median=baseline['raw_frame_median'],
                          p90=baseline['raw_frame_p90'],n=baseline['frames'])}
        with torch.inference_mode():
            for arm in ('point','direct','hough'):
                for seed in (1,2,3):
                    name=f'{arm}_seed{seed}';px=[];failure=0
                    for p in read(run/'predictions'/f'{name}_synth_val.json')['rows']:
                        if not p['pose_valid']:failure+=1;continue
                        O=observations[p['id']]
                        uv=G.project(O['K'],torch.tensor(p['R'])[None],torch.tensor(p['t'])[None],G.corners(O['dims']))[0][0,:8]
                        errors,_=set_error(uv.numpy(),truth[p['id']]);px.append(float(errors.mean()))
                    geom[name]=dict(mean=float(np.mean(px)) if px else None,
                        median=float(np.median(px)) if px else None,p90=float(np.quantile(px,.9)) if px else None,
                        n=len(px),pose_failures=failure)
        (run/'STRONG_BASELINE_COMPARISON.json').write_text(json.dumps(dict(
            scope='Same optimal one-to-one unordered corner-set metric for every arm. '
                  'Scoring-only geometry diagnostic; not the main fixed-ID metric.',results=geom),indent=2)+'\n')
        stock_backstop=all(geom[f'hough_seed{s}']['mean'] is not None and
            geom[f'hough_seed{s}']['mean']<=geom['stock']['mean'] and
            geom[f'hough_seed{s}']['p90']<=geom['stock']['p90'] and
            geom[f'hough_seed{s}']['pose_failures']==0 for s in (1,2,3))
        if hpass and not stock_backstop:
            status='NO_GO_STRONG_BASELINE'
            lines[2]=f'판정: **{status}**. 새 점 head 대비 합성 gate는 통과했지만 원래 YOLO 기하 성능을 보존하지 못했다.'
        lines+=['','## 원래 YOLO와 공간 배치 비교','',
            '양쪽 모두 평가에서만 최적 일대일 코너 대응을 허용했다. 번호 규약 차이를 제외한 '
            '공간 배치 진단이며 실제 번호 교정 성능이나 주 지표가 아니다.','',
            '| 모델 | 공간 배치 평균 px | 중앙값 px | P90 px | 유효 frame |',
            '|---|---:|---:|---:|---:|']
        for name,g in geom.items():
            vals=[f'{g[k]:.2f}' if g[k] is not None else '없음' for k in ('mean','median','p90')]
            lines.append('| '+name+' | '+' | '.join(vals)+f" | {g['n']} |")
    lines+=['','## 같은 점·같은 초기값에 선만 바꾼 진단','',
        '아래는 합성 calibration 253장의 원인 분리 결과다. 정답선과 GT 근처 초기값은 '
        '특권 정보이며 실제 모델 성능이 아니다. 역할 shuffle은 예측선 12개를 한 칸 순환한다.','',
        '| 구조/seed | 실제 초기값·선 없음 | 예측선 | 섞은 선 | 정답선 | GT근처·정답선 |',
        '|---|---:|---:|---:|---:|---:|']
    interventions={}
    strata={}
    for cell,c in diag['cells'].items():
        s=c['summary'];keys=['actual/none','actual/predicted','actual/shuffled','actual/oracle','near_gt/oracle']
        lines.append('| '+cell+' | '+' | '.join(f"{s[k]['primary']:.5f}" for k in keys)+' |')
        interventions[cell]=dict(predicted_better_than_no_lines=s['actual/predicted']['primary']<s['actual/none']['primary'],
            predicted_better_than_shuffled=s['actual/predicted']['primary']<s['actual/shuffled']['primary'],
            oracle_better_than_no_lines=s['actual/oracle']['primary']<s['actual/none']['primary'])
        by_source={k:{r['id']:r for r in v} for k,v in c['per_frame'].items()}
        strata[cell]={}
        for axis in ('view','height'):
            for value in sorted({v[axis] for v in diag['views'].values()}):
                ids=[fid for fid,v in diag['views'].items() if v[axis]==value]
                strata[cell][value]=dict(n=len(ids),**{source:float(np.mean([
                    by_source['actual/'+source][fid]['normalized'] for fid in ids]))
                    for source in ('none','predicted','shuffled','oracle')})
        for label,accept in [('easy_points',lambda r:r['valid'] and r['raw_px']<=10),
                             ('hard_points',lambda r:not r['valid'] or r['raw_px']>50)]:
            ids=[fid for fid,r in by_source['actual/none'].items() if accept(r)]
            strata[cell][label]=dict(n=len(ids),**{source:float(np.mean([
                by_source['actual/'+source][fid]['normalized'] for fid in ids])) if ids else None
                for source in ('none','predicted','shuffled','oracle')})
    (run/'CONDITION_BREAKDOWN.json').write_text(json.dumps(strata,ensure_ascii=False,indent=2)+'\n')
    lines+=['','## 구도별 진단 (Hough seed 1, 합성 calibration)','',
            '| 조건 | 장수 | 선 없음 | 예측선 | 정답선 |','|---|---:|---:|---:|---:|']
    for name,row in strata['hough_seed1'].items():
        vals=[f"{row[k]:.5f}" if row[k] is not None else '표본 없음' for k in ('none','predicted','oracle')]
        lines.append(f"| {name} | {row['n']} | "+' | '.join(vals)+' |')
    lines+=['','정면은 물체 기준 최근접 측면에 대해 10° 이내, 낮은 시점은 높이각 15° 이하로 사전 정의했다. '
            'GT 자세는 이 사후 조건 분류에만 사용했다. 조건별 표본이 적으면 확정 결론을 내리지 않는다.']
    line_summary={}
    for cell in diag['cells']:
        line_summary[cell]={}
        for kind in ('horizontal','height'):
            subset=[r for r in diag['line_metrics'] if r['cell']==cell and r['kind']==kind]
            line_summary[cell][kind]=dict(n_edges=len(subset),
                top1_median_normal_px=float(np.median([r['top1_normal_error_px'] for r in subset])),
                best_mode_median_normal_px=float(np.median([r['oracle_best_mode_error_px'] for r in subset])))
    (run/'LINE_ACCURACY.json').write_text(json.dumps(line_summary,ensure_ascii=False,indent=2)+'\n')
    trace_summary={}
    for arm in ('point','direct','hough'):
        for seed in (1,2,3):
            name=f'{arm}_seed{seed}'
            trace=[json.loads(x) for x in (run/'cache_heads'/name/'TRACE.jsonl').read_text().splitlines()]
            trace_summary[name]=dict(updates=len(trace),first100_mean_loss=float(np.mean([r['loss'] for r in trace[:100]])),
                previous200_mean_loss=float(np.mean([r['loss'] for r in trace[-400:-200]])),
                last200_mean_loss=float(np.mean([r['loss'] for r in trace[-200:]])),
                last200_point_loss=float(np.mean([r['point'] for r in trace[-200:]])),
                last200_pose_loss=float(np.mean([r['pose'] for r in trace[-200:]])),
                note='Loss magnitude differs by arm because line loss is absent in point-only. '
                     'A finite, falling loss alone does not certify convergence or accuracy.')
    (run/'TRAINING_TREND.json').write_text(json.dumps(trace_summary,indent=2)+'\n')
    lines+=['','## 한계와 다음 결정','',
        '- 정답선으로 좋아지는 경우: 결합 경로에는 활용 가능한 여지가 있다. 예측선이 실패하면 선 관측/역할/신뢰도 또는 공동학습을 점검한다.',
        '- 정답선도 실제 초기값에서는 실패하고 GT 근처에서만 좋아지는 경우: 초기화/유효 후보/국소 최적화가 병목이다.',
        '- 선을 제거하거나 섞어도 개선이 유지되면 선 증거의 추가 이득으로 주장하지 않는다.',
        '- 합성 gate 실패는 이 구조·예산에서의 중단 결정이다. 모든 점·선 아키텍처의 불가능성을 입증하지 않는다.',
        '- 점 전용 새 head 자체가 충분히 학습됐는지, 원래 강한 YOLO보다 좋은지도 별도 조건이다. 약한 새 baseline만 이겼다고 채택하지 않는다.',
        '- 실사 forward/평가, 원본 manual GT 교정, FINAL 개봉, self-training은 수행하지 않았다.',
        '- C4가 확인된 asset이 없으므로 C1/C2만 평가했다. 캐시의 instance는 GT IoU로 평가 매칭된 조건이라는 한계가 있다.',
        '- synthetic frame의 원본 이미지 SHA 분리는 직접 검증했다. target 재투영 0px는 export 내부 일치이며 원래 GT 전체의 독립 인증은 아니다.','',
        '재현: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u '
        'scripts/research/pallet_point_line_decisive_v6/cli_driver.py` (완료 cell을 재사용하며 동결 소스 SHA를 검증).']
    interpretation=run/'INTERPRETATION_KO.md'
    if interpretation.exists():
        lines += ['', interpretation.read_text()]
    (run/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    decision=dict(status=status,synthetic_hough_gate=hpass,synthetic_direct_gate=dpass,
        original_stock_geometry_backstop=stock_backstop,
        hough_specific_gate=comparisons['H_VS_D']['numeric_gate_passed'],interventions=interventions,
        real_performance_established=False,universal_impossibility_established=False,
        gallery=gallery,training_steps=18000,seeds=[1,2,3])
    (run/'DECISION.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    # Plain preformatted report keeps the exact source numbers and contract
    # available beside a selectable image gallery without a JS dependency.
    options=''.join(f'<option value="{html.escape(g["path"])}">{html.escape(g["id"])} · {g["selection"]}</option>' for g in gallery)
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>점·선 융합 최종 파일럿</title><style>body{{max-width:1400px;margin:32px auto;padding:0 22px;font-family:system-ui,sans-serif;background:#101725;color:#e7edf5}}pre{{white-space:pre-wrap;line-height:1.65;font-size:15px}}img{{width:100%;background:white;border-radius:8px}}select{{padding:10px;width:100%;margin:12px 0}}a{{color:#87c7ff}}details{{margin:22px 0}}</style>
<h1>점·선 융합: {status}</h1><p>합성 512장 · 3구조 × 3seed · 실사 성능 미확립</p>
<label>예측 비교<select onchange="document.getElementById('gallery').src=this.value">{options}</select></label>
<img id="gallery" src="{gallery[0]['path']}" alt="원본, 점 전용, Hough 공동 자세, 예측선 비교">
<p>청록색: 합성 GT. 자세 표시는 평가에 허용된 C2 하나로 정렬했습니다. 선은 실제 highest-mass 예측입니다. attention/가시성 정답을 나타내지 않습니다.</p>
<pre>{html.escape(chr(10).join(lines))}</pre>
<p><a href="DECISION.json">판정 JSON</a> · <a href="DIAGNOSTICS.json">전체 원인 진단</a> · <a href="PROTOCOL_LOCK.json">고정 프로토콜</a> · <a href="REPORT_KO.md">보고서 원문</a></p></html>'''
    (run/'index.html').write_text(body)
    try:
        attempt=subprocess.Popen(['xdg-open',str((run/'index.html').resolve())],
                                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        opened=dict(attempted=True,pid=attempt.pid,verified_window_open=False,
                    note='Opening requested immediately after HTML creation; root must verify actual visible window.')
    except OSError as exc:
        opened=dict(attempted=False,error_type=type(exc).__name__,verified_window_open=False)
    (run/'BROWSER_OPEN_ATTEMPT.json').write_text(json.dumps(opened,indent=2)+'\n')
    print(json.dumps(dict(status=status,report=str(run/'index.html'),gallery=len(gallery)),ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);main(ap.parse_args())
