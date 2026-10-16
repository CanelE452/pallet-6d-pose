"""Render measured C1/C4 trial outcomes and verify visible desktop delivery."""
from pathlib import Path
import json,sys,re,subprocess,time,hashlib
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import numpy as np
import markdown
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pallet_c4_joint_loss_v1 as exp

ROOT=exp.ROOT;OUT=exp.OUT;NOTE=ROOT/'_docs/notes/pose_geometry/c4-joint-loss-check.md'
TITLE='C1 보존과 C4 전체 Loss — 실제 학습 비교'

def main():
    audit=exp.read(OUT/'FINAL_AUDIT.json');assert audit['PASS']
    s=exp.read(OUT/'SUMMARY.json');d=exp.read(OUT/'DIAGNOSIS.json');pre=exp.read(OUT/'PREFLIGHT.json')
    labels={'stock':'학습 전 YOLO','indexed':'일반 loss 추가 학습','point_select':'점 loss로 대칭 선택','joint_select':'전체 loss로 대칭 선택'}
    def stats(arm,pop,key):
        return [s['results'][arm][pop][key]] if arm=='stock' else [s['results'][f'{arm}_s{i}'][pop][key] for i in [1,2,3]]
    table='| 방법 | C4 평균 px | C4 중앙값 px | C4 P90 px | C1 평균 px | C1 P90 px |\n|---|---:|---:|---:|---:|---:|\n'
    expected=[]
    for arm,label in labels.items():
        vals=[np.mean(stats(arm,pop,k)) for pop,k in [('C4','matched_mean_px'),('C4','matched_median_px'),('C4','matched_p90_px'),('C1','matched_mean_px'),('C1','matched_p90_px')]]
        cells=[label]+[f'{v:.4f}' for v in vals];expected.extend(cells);table+='| '+' | '.join(cells)+' |\n'
    identical=all(v['point_vs_joint_max_abs']==0 for v in d['state_differences'].values())
    pred=np.load(OUT/'predictions.npz')
    diffs=[float(np.max(np.abs(pred[f'joint_select_s{i}']-pred[f'point_select_s{i}']))) for i in [1,2,3]]
    switches={a:[d['training_history'][f'{a}_s{i}']['history'][-1]['choice_disagreements'] for i in [1,2,3]] for a in exp.ARMS}
    conclusion=('**전체 loss로 선택하도록 바꿔도 기존 점 loss 선택보다 성능이 나빠지지 않았다. 세 시드에서 최종 가중치와 예측이 정확히 같았다.**'
                if identical and not any(diffs) else '**전체 loss 선택의 추가 효과는 아래 실제 시드별 지표로 판단한다. 단순한 최소값 성질이 학습 후 비악화를 보장하지는 않는다.**')
    contrasts='| 비교 | 대상 | 주 지표 차이 seed1 / 2 / 3 | 좋음→큰 오류 seed1 / 2 / 3 |\n|---|---|---|---|\n'
    for control in ['point_select','indexed','stock']:
        for pop in ['C4','C1']:
            c=s['comparisons']['joint_minus_'+control][pop]
            contrasts+='| 전체 선택 − '+labels[control]+' | '+pop+' | '+' / '.join(f'{x:+.8f}' for x in c['primary_delta_per_seed'])+' | '+' / '.join(map(str,c['new_good_to_bad_per_seed']))+' |\n'
    fits='| 모델 | C4 학습 전체 loss | C1 학습 전체 loss |\n|---|---:|---:|\n'
    for arm,label in labels.items():
        names=['stock'] if arm=='stock' else [f'{arm}_s{i}' for i in [1,2,3]]
        values=[np.mean([d['fit'][n]['train'][pop]['mean_joint_loss'] for n in names]) for pop in ['C4','C1']]
        fits+='| '+label+' | '+' | '.join(f'{v:.5f}' for v in values)+' |\n'
    cache=exp.read(OUT/'CACHE_COMPLETE.json')['splits'];r4=s['results']['stock']['C4'];r1=s['results']['stock']['C1']
    text=f'''# C1은 기존 loss 유지, C4는 전체 loss 최소 선택 — 실제 검증

2026-09-11. {conclusion}

**단, 학습 전 YOLO 대비 C1 평균 오차는 4.607→4.988px로 악화했다. C4는 4.970→3.132px로 개선했다. 아래 비악화 판정은 추가 학습한 대조군 대비이며, 학습 전 모델 대비 모든 대상이 보존되었다는 뜻이 아니다.**

[HTML](../../../data/pallet/results/pallet_c4_joint_loss_v1/index.html) · [수치 JSON](../../../data/pallet/results/pallet_c4_joint_loss_v1/SUMMARY.json) · [코드](../../../scripts/research/pallet_c4_joint_loss_v1.py)

## 질문과 실행 범위

사용자 질문은 “대칭 대상만 loss를 바꾸고 나머지는 기존 loss를 유지하면 성능이 나빠지는가?”였다. 일반 loss, 점 위치 loss로만 대칭을 선택하는 방식, 점·가시성·RLE 전체 합계로 선택하는 방식의 세 군을 같은 초기화·배치 순서·예산으로 비교했다.

**9개 모델, 총9,000 optimizer updates를 실제 완료했다.** 군별3시드·1,000업데이트, batch32, AdamW lr0.0001/weight decay0.0001. 기존 YOLO의 one2one 점·sigma 출력층과 같은 초기 FiLM만 학습했다. 치수는 모든 군에 동일하게 제공했고, 백본·검출기·공간 특징·BN·flow는 동결했다. 현재 후보별 ±16px 보정 아키텍처와 결합한 실험이나 전체 YOLO 재학습은 아니다.

학습 C1 {cache['train']['C1']}개·실물 C4 {cache['train']['C4']}개. 평가 C1 {r1['n']}장·C4 {r4['n']}장, 공통 검출·매칭은 각각 {r1['matched']}/{r1['n']}, {r4['matched']}/{r4['n']}장이다. C1은 기존 합성 데이터에서 대칭을 허용하지 않았던 대상, C4는 등록된110×110cm 정사각 실물이다. 실물 평가155장은 이전에 사용한 interleave DEV이며 새 촬영 세션 일반화가 아니다. C1과 C4는 도메인도 다르므로 둘 사이 성능 차이를 대칭 유무의 효과로 해석하지 않는다.

## 정확히 바꾼 loss

```text
C1: L = L_identity                          # 세 군 모두 동일
C4 일반: L = L_identity
C4 기존 선택: g* = argmin_g L_point(g)
C4 전체 선택: g* = argmin_g [12 L_point(g) + L_visibility(g) + L_RLE(g)]
최종 학습: 선택한 하나의 g*로 점·가시성·RLE를 함께 계산
```

한 학습 객체에 TAL anchor 하나를 사용한다. 세 군 모두 동일한 객체별 reduction과 객체별 RLE clamp를 쓴다. 이는 원래 YOLO의 여러 객체를 한 번에 합친 batch-global RLE clamp를 그대로 패치한 실험이 아니다. **선택 기준의 차이를 비교하는 통제 시험**이며 stock 전체 학습과 동일하다고 주장하지 않는다. C1에서 기존 객체별 loss와 gradient가 바뀌지 않는 것을 직접 검사했다.

## 결과

{table}

매칭 대상의 valid8코너 오차. 학습군은3시드 통계의 평균이며, C4는 모든 군에서 동일한 네 회전을 인정한다. C1은 번호를 그대로 유지한다. 전체 프레임 주 지표는 미검출에 이미지 대각선 벌점을 주고 대각선으로 정규화한다. 최고 시드나 checkpoint를 고르지 않았다.

![C1과 C4의 실제 비교](../../../data/pallet/results/pallet_c4_joint_loss_v1/comparison.png)

{contrasts}

주 지표 차이는 음수가 개선, 양수가 악화다. 좋음→큰 오류는 대조군≤10px에서 전체 선택>20px로 바뀐 장면 수이며 검출 실패도 포함한다. 사전 비악화 판정: **{s['decision']}**. 이 판정은 기존 점 선택뿐 아니라 일반 loss 대조군 대비 C1/C4 보호와 C4 P90까지 요구한다. 따라서 기존 점 선택과 같다는 사실만으로 모든 판정 조건이 충족되는 것은 아니다.

## 실제로 선택 기준 차이가 발생했나

- 학습 전 실물 C4 {pre['initial']['C4']}개에서 점 선택과 전체 선택이 서로 다른 후보를 고른 경우: **{pre['initial']['disagreement']}개**.
- 학습 전 두 기준의 C4 평균 전체 loss 차이: **{pre['mean_full_loss_drop_C4']:.8f}**.
- 전체 학습의 후보 기준 불일치 누적 횟수: 일반군 {switches['indexed']}, 점 선택군 {switches['point_select']}, 전체 선택군 {switches['joint_select']}. 매 업데이트의 반복 관측 횟수이며 고유 이미지 수가 아니다.
- 점 선택과 전체 선택의 최종 모델 파라미터 최대 차이: {[v['point_vs_joint_max_abs'] for v in d['state_differences'].values()]}.
- 같은 평가 이미지의 두 방식 예측 최대 차이(px): **{diffs}**.

{'**앞서 지적한 “점 loss만으로 후보를 고르는 구현”은 이 데이터·초기화·학습 범위에서는 실제 후보 차이를 만들지 않았다. 따라서 그것을 이번 성능 악화의 원인이라고 할 수 없다.** 전체 loss로 고치는 것은 목적과 구현을 일치시키지만, 선택이 같으면 gradient와 학습 결과도 같다.' if identical else '후보 기준 불일치가 있더라도 그것만으로 성능 개선/악화를 설명할 수 없다. 위의 동일 예측 검사와 실제 학습 후 지표를 함께 읽는다.'}

## loss 감소와 정확도는 구분

{fits}

위 loss는 같은 객체별 전체 loss로 재계산한 학습 적합도다. loss에는 sigma·RLE·가시성이 있고 위치 평가는 raw8점 거리이므로 수치의 변화가 일대일 대응하지 않는다. 학습은 TAL anchor, 평가는 GT 없이 고른 최고 confidence 검출이므로 train/val 적합도 차이를 오직 과적합의 인과 증거로 해석하지 않는다.

실제로 전체 선택 seed1의 **같은 C1 학습 대상**에서 전체 loss는 0.96681→0.87497로 감소했지만 평균 위치 오차는 6.033→6.166px로 증가했다. seed2·3에서도 loss는 감소하고 위치 오차는 증가했다. seed1의 평균 sigma는 0.521→0.623으로 증가했다. 이는 학습 목적(영역 정규화된 OKS·RLE·가시성)과 raw 픽셀 지표가 함께 좋아지지 않을 수 있음을 보여준다. sigma 증가 자체가 유일한 원인이라는 증거는 아니며, 공유 가중치 간섭이나 도메인 적응의 기여도는 별도 통제가 필요하다.

일반 loss 추가 학습의 C1 평균 오차도 5.045px로 악화했다. 따라서 이번 C1 악화를 C4 전체 선택으로 변경한 효과라고 해석하면 안 된다. 전체 선택의 C4 개선량은 일반 추가 학습보다 약0.031px이고, 세션 단위 bootstrap 신뢰구간은 0을 포함해 추가 이득이 확립된 것은 아니다.

C1의 **loss 식과 같은 예측에서의 gradient는 동일**해도, C4 학습을 통해 공유 출력층 가중치가 바뀌면 C1 예측이 달라질 수 있다. 실제 C1 변화량은 위 표에 제시했다. 이번 실험에서 gradient 충돌을 원인으로 확정한 것은 아니다.

## 검증과 해석 한계

- C1 loss/gradient 동일, 전체 loss의 identity·점 선택 대비 비증가, 네 후보 직접 재계산, C4 회전 대응 불변성 검사 통과.
- 모델 출력 재현 최대 차이 {audit['model_replay_max_px']:.8f}px, 지표 독립 재계산 최대 차이 {audit['metric_replay_max_px']:.8f}px.
- 원본 checkpoint, C1 캐시, 실물 이미지와 수작업 라벨 hash 보존. 이번 검사는 좌표 전처리 왕복과 같은 GT 사용을 확인한 것이며 수작업 annotation 자체를 독립 재작성한 감사가 아니다.
- 원래 multi-anchor 전체 네트워크 학습, 실제 방향성 C1 팔레트, 새 세션·새 종류에 대한 결과는 아니다. 이 제한된 비교를 일반적인 성능 비악화 보장으로 확대하지 않는다.

## 산출물

[프로토콜](../../../data/pallet/results/pallet_c4_joint_loss_v1/PROTOCOL.json) · [학습 완료9모델](../../../data/pallet/results/pallet_c4_joint_loss_v1/TRAIN_COMPLETE.json) · [사후 진단](../../../data/pallet/results/pallet_c4_joint_loss_v1/DIAGNOSIS.json) · [최종 감사](../../../data/pallet/results/pallet_c4_joint_loss_v1/FINAL_AUDIT.json)
'''
    NOTE.write_text(text)
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'svg.fonttype':'path'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,pop in zip(axes,['C4','C1']):
        for j,arm in enumerate(labels):
            vals=stats(arm,pop,'matched_mean_px');ax.bar(j,np.mean(vals),color=['#8292a4','#a69ac9','#d49a42','#21838a'][j],width=.6)
            if len(vals)>1:ax.scatter([j]*len(vals),vals,color='#243746',s=18,zorder=3)
            ax.text(j,max(vals)+.03,f'{np.mean(vals):.3f}',ha='center',va='bottom',fontsize=10)
        ax.set_xticks(range(4),['학습 전','일반 loss','점 선택','전체 선택']);ax.set_title(pop+(' · 정사각 실물' if pop=='C4' else ' · 기존 loss 유지 대상'))
        ax.set_ylabel('매칭 대상 평균 코너 오차 px ↓');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.margins(y=.18)
    fig.suptitle('동일 입력·출력층·예산 / 3시드 실제 학습');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.svg');plt.close(fig)
    def link(m):
        ref=m[1]
        if '://' in ref or ref.startswith('#'):return m[0]
        p=(NOTE.parent/ref).resolve();assert p.exists() or p==OUT/'index.html',p
        return ']('+p.as_uri()+')'
    body=markdown.markdown(re.sub(r'\]\(([^)]+)\)',link,text),extensions=['tables','fenced_code'])
    css='body{font:16px/1.85 system-ui,sans-serif;color:#21334a;background:#edf2f7;margin:0}main{max-width:1240px;margin:25px auto;padding:35px 45px;background:white;border-radius:16px}h1{font-size:29px}h2{margin-top:38px;border-bottom:2px solid #e0e9f0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#eaf1f8}img{max-width:100%}pre{background:#19334b;color:white;padding:18px;border-radius:8px;overflow:auto}a{color:#16699c}strong{color:#116c79}'
    page=f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>{TITLE}</title><style>{css}</style><main>{body}</main></html>'
    (OUT/'index.html').write_text(page)
    class Check(HTMLParser):
        def __init__(self):super().__init__();self.active=False;self.cells=[];self.buf=''
        def handle_starttag(self,t,a):
            if t in ['td','th']:self.active=True;self.buf=''
            for k,v in a:
                u=urlparse(v)
                if k in ['src','href'] and u.scheme=='file':assert Path(unquote(u.path)).exists()
        def handle_data(self,s):
            if self.active:self.buf+=s
        def handle_endtag(self,t):
            if t in ['td','th']:self.cells.append(self.buf.strip());self.active=False
    parser=Check();parser.feed(page);assert parser.cells[6:30]==expected
    url=(OUT/'index.html').as_uri()
    with (OUT/'browser.log').open('w') as log,(OUT/'rendered_dom.html').open('w') as dom:
        subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run',
            '--user-data-dir=/tmp/pallet-joint-loss-chrome','--window-size=1550,1450','--screenshot='+str(OUT/'browser_preview.png'),'--dump-dom',url],stdout=dom,stderr=log,check=True,timeout=40)
    p=Check();p.feed((OUT/'rendered_dom.html').read_text());assert p.cells[6:30]==expected
    subprocess.run(['google-chrome','--new-window',url],capture_output=True,check=True,timeout=20)
    windows=[]
    for _ in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if TITLE in title:
                info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout
                windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    visible=any(w['viewable'] for w in windows);exp.write(OUT/'DELIVERY.json',dict(html_open_confirmed=visible,windows=windows,table_verified=True));assert visible
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['COMPLETION.json','DISCORD_NOTIFICATION.json'] and p.suffix!='.log']
    files += [NOTE,Path(__file__),ROOT/'scripts/research/pallet_c4_joint_loss_v1.py',ROOT/'scripts/research/pallet_c4_joint_loss_diagnose.py']
    exp.write(OUT/'COMPLETION.json',dict(complete=True,models=9,optimizer_steps=9000,html_open_confirmed=True,
        point_vs_joint_identical=identical,decision=s['decision'],files={str(p.relative_to(ROOT)):exp.sha(p) for p in files}))
    print('Report, table and visible Chrome verified.')

if __name__=='__main__':main()
