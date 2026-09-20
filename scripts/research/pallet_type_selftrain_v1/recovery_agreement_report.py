"""Report fixed pseudo-mask experiment, not a new independent test."""
from pathlib import Path
import numpy as np
from . import recovery_agreement as A
from scripts.research.pallet_dim_conditioned_p_v1.eval_math import summary
from scripts.research.pallet_posefix_large_error_v1.evaluate import recovery_damage

C=A.C;R=A.R


def main():
    lock=C.read(A.DOC/'OUTPUTS_LOCK.json')
    for b in lock['artifacts']:C.verify(b)
    records=[r for r in C.read(R.BASE_DOC/'EVAL_PROTOCOL.json')['records'] if r['kind']=='PLASTIC']
    ids=[r['id'] for r in records]
    assert len(ids)==194
    results_paths={arm:A.RAW/f'SCREEN_{arm}.json' for arm in ['AGREE','COUNT']}
    results_paths['REF_LR5']=R.RAW/'pose_only/SCREEN_REF_LR5.json'
    metrics={k:{r['id']:r for r in C.read(p)['metrics']} for k,p in results_paths.items()}
    metrics['R0']={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    for rows in metrics.values():assert set(rows)==set(ids)
    subset_path=R.BASE_DOC/'large_corner_recovery_v1/extreme_low_subset_v1/SUBSET_MANIFEST.json'
    retained={r['id'] for r in C.read(subset_path)['records'] if r['kind']=='PLASTIC'}
    assert len(retained)==159 and retained<=set(ids)
    def pack(keys):
        base=[metrics['R0'][i] for i in keys];out={}
        for arm,lookup in metrics.items():
            rows=[lookup[i] for i in keys]
            recovery=recovery_damage(base,rows,True)
            recovery_frames=[]
            for b,n in zip(base,rows):
                if b['matched'] and any(x is not None and x>20 and y<=10 for x,y in zip(b['canonical_errors'],n['canonical_errors'])):
                    recovery_frames.append(b['id'])
            out[arm]=dict(summary=summary(rows),recovery=recovery,recovery_frames=recovery_frames,
                all_frame_recovery=recovery_damage(base,rows,False))
        return out
    full=pack(ids);restricted=pack([i for i in ids if i in retained])
    checks={}
    for arm in ['AGREE','COUNT']:
        d=full[arm];s=d['summary'];r=d['recovery']
        checks[arm]=dict(recovered_at_least5=r['recovered']>=5,
            recovery_in_at_least3_frames=len(d['recovery_frames'])>=3,
            damage_rate_le1percent=r['damage_rate'] is not None and r['damage_rate']<=.01,
            PCK20_at_least_R0=s['PCK']['20']>=full['R0']['summary']['PCK']['20'])
    mask=C.read(A.DOC/'MASK_COMPLETE.json')
    paths=(C.ROOT/C.read(A.DOC/'PROTOCOL.json')['datasets']['AGREE']['train_list']['path']).read_text().splitlines()[512:]
    sampled=sorted({Path(p).stem for p in paths});assert len(sampled)==217
    exposure={}
    for arm,key in [('AGREE','agreement'),('COUNT','count_control'),('REF_LR5','original_valid')]:
        rr=[C.read(A.RAW/'masks'/f'{i}.json') for i in sampled]
        exposure[arm]=dict(unique_images=len(rr),corners=sum(sum(r[key][:8]) for r in rr),
            corrections_over20=sum(sum(v and d>20 for v,d in zip(r[key][:8],r['raw_to_refined_shift_px'][:8])) for r in rr))
    # Verify every surviving target remains byte-token identical to the REF source.
    old_list=(C.ROOT/C.read(R.DOC/'pose_only/PROTOCOL.json')['datasets']['REF']['train_list']['path']).read_text().splitlines()
    for arm in ['AGREE','COUNT']:
        new_list=(C.ROOT/C.read(A.DOC/'PROTOCOL.json')['datasets'][arm]['train_list']['path']).read_text().splitlines()
        assert [Path(p).name for p in new_list]==[Path(p).name for p in old_list]
        for p,q in zip(map(Path,old_list),map(Path,new_list)):
            before=(p.parent.parent/'labels'/p.with_suffix('.txt').name).read_text().split()
            after=(q.parent.parent/'labels'/q.with_suffix('.txt').name).read_text().split()
            assert before[:5]==after[:5]
            if len(after)==32:
                for i in range(9):
                    if float(after[7+3*i])==2:assert before[5+3*i:8+3*i]==after[5+3*i:8+3*i]
    for arm in ['AGREE','COUNT']:
        fit=C.read(A.DOC/f'FIT_{arm}.json');C.verify(fit['checkpoint'])
        assert fit['optimizer_steps']==320 and fit['protected_state_exact']
    result=dict(full194=full,retained159=restricted,checks=checks,
        gate_pass={a:all(v.values()) for a,v in checks.items()},pool_mask_totals=mask['totals'],sampled_unique=exposure,
        coordinates_and_boxes_unchanged=True,same_images_and_order=True,new_annotations=0,new_tags=0,
        auto_promoted=False,independent_confirmation=False,goal_complete=False,
        sources=[C.bound(p) for p in results_paths.values()]+[C.bound(A.DOC/'OUTPUTS_LOCK.json'),C.bound(subset_path),C.bound(__file__)])
    C.freeze(A.DOC/'RESULTS.json',result)
    lines=['# 추가 어노테이션 없는 self-training — 두 학생 동의 마스크', '',
        '새 수동 정답0·태그 라벨0. 기존 보정 좌표와 이미지 순서·횟수는 유지했다. AGREE는 두 학생이 기존 보정 좌표에 동의하는 코너만 학습하고, COUNT는 동일 이미지에서 같은 수의 코너를 무작위로 남긴다. 두 새 학생은 R0에서 시작해320step씩 학습했다.', '',
        '## 동일 corner8 비교', '',
        '| 평가 | 모델 | 중앙 px | P90 px | PCK20 % | >20→≤10 복구 | <5→>10 손상 | 복구 이미지 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for label,table in [('전체194',full),('극저각 제외159',restricted)]:
        for arm in ['R0','REF_LR5','AGREE','COUNT']:
            x=table[arm];s=x['summary'];d=x['recovery']
            lines.append(f'| {label} | {arm} | {s["matched_pooled_corner8_median_px"]:.3f} | {s["matched_pooled_corner8_P90_px"]:.3f} | {100*s["PCK"]["20"]:.2f} | {d["recovered"]}/{d["hard"]} | {d["damaged"]}/{d["good"]} | {len(x["recovery_frames"])} |')
    lines+=['',f'사전 고정 전체194 screen 통과: {result["gate_pass"]}. 최종 모델은 교체하지 않았다.', '',
        '복구·손상은 매칭된 동일 canonical GT 코너 기준이다. 검출 실패를 포함한 수치도 RESULTS.json에 보존한다. PCK20에는 매칭 실패 패널티가 포함된다.', '',
        '## 학습 신호의 제한', '',
        f'249장 후보 코너: {mask["totals"]}',f'실제 사용217장 고유 코너: {exposure}', '',
        '- 동의 마스크가 원래20px 넘는 보정을 모두 제외했다. 동의는 정확성 증명이 아니며 이 방식은 큰 보정 신호를 약화시킬 수 있다.',
        '- 두 교사는 같은 초기 모델과 수도레이블 풀에 의존한다. 독립 교사로 간주하지 않는다.',
        '- 기존 Replay는 과거 실사 수동 감독을 사용했고 기존 DEV194에 교사 학습과 겹치는3장이 있다. 여기서 새 정답을 쓰지 않았다는 뜻이지 과거 감독이 전혀 없다는 뜻은 아니다.',
        '- 반복 사용한 DEV이며, 새 독립 시험이나 통계적 확증은 아니다. 제외159는 사후 범위 제한으로 전체194와 함께 기록했다.',
        '- 기준 통과 여부와 별개로 이 실험만으로 전체 큰 코너 복구 목표를 완료 처리하지 않는다.', '']
    C.write_text(A.RAW/'REPORT_KO.md','\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
