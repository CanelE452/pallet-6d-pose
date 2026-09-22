"""Local-only report extras after the fixed Clean19 pilot has completed."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scripts.research import pallet_replay_clean19_v1 as P


def main():
    assert P.read(P.DOC/'AUDIT.json')['complete']
    result=subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).with_name('test_pallet_replay_clean19_v1.py'))],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    P.save(P.DOC/'TEST_RESULTS.json',dict(PASS=True,stdout=result.stdout,stderr=result.stderr))
    s=P.read(P.DOC/'RESULTS.json');m=P.read(P.RAW/'FRAME_METRICS.json');records=P.read(P.DOC/'SPLIT.json')['evaluation']
    # Failure retention and partition arithmetic, independent of training code.
    for arm in P.ARMS:
        assert len(m[arm])==300
        for field in ('corners','correct10','gross20_count','total_frames'):
            assert sum(s[c][arm][field] for c in P.SEVERITIES)==s['ALL300'][arm][field]
        for fid,row in m[arm].items():
            assert row['matched']==m['R0'][fid]['matched']
            assert row['canonical_valid']==m['R0'][fid]['canonical_valid']
            if not row['matched']:assert all(e==800 for e in row.get('errors',[]))
    group_ids=sorted({r['recording'] for r in records});cluster={g:j for j,g in enumerate(group_ids)}
    ci={}
    for baseline in ('PRIOR1','N2_DIM_ONLY','N3_DIM_SYM','R0'):
        numerator=np.zeros(len(group_ids));denom=np.zeros(len(group_ids))
        for r in records:
            fid=r['id'];a=m['CLEAN19_REPLAY'][fid];b=m[baseline][fid];j=cluster[r['recording']]
            numerator[j]+=sum(e<=10 for e in a.get('errors',[]))-sum(e<=10 for e in b.get('errors',[]))
            denom[j]+=a.get('corners',0)
        rng=np.random.default_rng(20260922);samples=[]
        for _ in range(5000):
            index=rng.integers(0,len(group_ids),len(group_ids));d=denom[index].sum()
            if d:samples.append(100*numerator[index].sum()/d)
        ci[baseline]=dict(delta_pp=float(100*numerator.sum()/denom.sum()),CI95_pp=np.quantile(samples,[.025,.975]).tolist(),
            units='source recording clusters',clusters=len(group_ids),resamples=5000,
            exploratory=True,multiplicity_adjusted=False,same_session_adaptation=True)
    P.save(P.DOC/'PAIRED_CLUSTER_UNCERTAINTY.json',ci)
    # Code-native SVG bar chart, linked in Markdown; no generated photograph.
    plot=['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="450" viewBox="0 0 1080 450"><rect width="1080" height="450" fill="#142631"/><g font-family="sans-serif" fill="white"><text x="24" y="30" font-size="23">Clean19 + synthetic replay — same 300 evaluation images</text><text x="24" y="55" font-size="15">PCK10 (%) against existing reference; same-session adaptation, not unseen-environment validation</text>']
    colors=['#aab9c2','#73bfff','#60dbab','#d3b4ff','#ffcf62','#ef927f']
    for j,arm in enumerate(P.ARMS):
        y=85+j*52;value=100*s['ALL300'][arm]['PCK']['10']
        plot.append(f'<text x="24" y="{y+24}" font-size="16">{arm}</text><rect x="260" y="{y}" width="{value*7}" height="33" fill="{colors[j]}"/><text x="{270+value*7}" y="{y+23}" font-size="17">{value:.2f}%</text>')
    plot.append('</g></svg>')
    P.save(P.DOC/'figures/PCK10.svg',''.join(plot))
    lines=['# Clean19/300 최종 요약','', '![동일300장 PCK10 비교](figures/PCK10.svg)','',
           '같은 촬영 세션을 허용한 소량 수동지도 적응. 학습19장·87개 직접클릭 코너, 평가300장. 데이터319장 중 실제 이미지 교차0. 사전근접중복검사 통과는 새로운 환경 독립성을 뜻하지 않는다.', '',
           '| 대비 기준 | 새 RAW Replay PCK10 차이(pp) | 탐색적 촬영그룹 bootstrap 95% CI |','|---|---:|---|']
    for a,c in ci.items():lines.append(f'| {a} | {c["delta_pp"]:+.2f} | [{c["CI95_pp"][0]:+.2f}, {c["CI95_pp"][1]:+.2f}] |')
    lines+=['','단일seed·반복개발평가·소수촬영그룹이므로 CI는 참고용이고 확정적 유의성/새환경 일반화 주장이 아니다. cap8은 사전등록 보조출력이며 평가 후 더 좋은 것을 골라 최종 모델로 바꾸지 않았다.', '',
            '[전체 수치](RESULTS_KO.md)', '', '로컬 비교 이미지: outputs/pallet_replay_clean19_v1/GALLERY.html', '',
            '원본·기존모델·기존319장표 보존. 신규 모델 자동 승격 없음. 추가 학습·하이퍼파라미터 탐색 없음. commit/push 없음.']
    P.save(P.DOC/'SUMMARY_KO.md','\n'.join(lines)+'\n')
    P.verify();print(result.stdout);print(json.dumps(ci,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
