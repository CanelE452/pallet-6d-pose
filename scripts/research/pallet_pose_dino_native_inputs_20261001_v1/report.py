"""Render verified input evidence without targets, weights or new inference."""
from pathlib import Path
import json
import numpy as np
from scripts.research.pallet_pose_dino_input_audit_20261001_v1.report import bind,read,verify

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments/pallet_pose_dino_native_inputs_20261001_v1'
OLD=ROOT/'_docs/experiments/pallet_pose_dino_input_audit_20261001_v1'

def main():
    import matplotlib.pyplot as plt
    r=read(DOC/'TRAIN_APPEARANCE_INPUTS.json');v=read(DOC/'INPUT_VERIFICATION.json')
    assert v['complete'] and v['PASS'] and v['public_input_receipt']==bind(DOC/'TRAIN_APPEARANCE_INPUTS.json')
    assert r['protocol']==v['protocol'] and r['descriptors']==v['descriptors']
    for b in (v['code'],v['protocol'],v['public_input_receipt']):verify(b)
    assert v['original_candidate_validity_unchanged'] and v['exact_removed_support_matches_parent_padding_audit']
    models=list(r['models']);x=np.arange(len(models));fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    before=[r['models'][m]['previous_supported_points'] for m in models]
    after=[r['models'][m]['native_supported_points'] for m in models]
    ax.bar(x-.2,before,.4,label='Prepared image support',color='#64748b')
    ax.bar(x+.2,after,.4,label='Original image support',color='#0d9488')
    for j,(a,b) in enumerate(zip(before,after)):
        ax.text(j-.2,a,str(a),ha='center',va='bottom');ax.text(j+.2,b,str(b),ha='center',va='bottom')
    ax.set_xticks(x,models);ax.set_ylim(0,max(before)*1.18);ax.set_ylabel('Sampled projected corner positions')
    ax.set_title('TRAIN input correction only | 2,598 frames / 20,776 valid candidates retained\nExclude reflected padding positions; reuse identical frozen image tokens')
    ax.legend(loc='lower right');ax.grid(axis='y',alpha=.2)
    folder=DOC/'figures';folder.mkdir(exist_ok=True)
    path=folder/'native_support_comparison.png';assert not path.exists();fig.savefig(path,dpi=150);plt.close(fig)
    lines=['# 원본 이미지 영역의 특징 검산 결과','',
        '**입력 검산 PASS. 이 단계에서 새 학습과 T/R 성능 평가는 하지 않았다.** 기존 합성 TRAIN 2,598행과 실패 1행, 유효 후보 20,776개를 유지했다. 반사 패딩 위치만 표본에서 제외했다.','',
        '| 후보 | 기존 지원점 | 원본 영역 지원점 | 제외한 패딩점 | 특징이 바뀐 후보 |',
        '|---|---:|---:|---:|---:|']
    for m in models:
        z=r['models'][m];lines.append(f"| {m} | {z['previous_supported_points']} | {z['native_supported_points']} | {z['removed_prepared_padding_points']} | {z['valid_descriptor_changed_candidates']} |")
    lines += ['', '![원본 영역 지원점 비교](figures/native_support_comparison.png)', '',
        '지원점은 고정 pose와 실제 W/H/D 치수, 기존 K로 투영한 8개 꼭짓점 중 양의 깊이·crop 범위·원본 영상 범위를 만족하는 점이다. 가시성 정답이 아니다. 원본 영상의 반열린 범위는 준비된 source 좌표에서 `[100, width−100) × [100, height−100)`이다. 실제 사진에는 padding을 추가하지 않는다.', '',
        '기존 4.69GB token을 그대로 재사용했다. 새 GPU forward·학습·PnP·정답 조회는 0회다. FP32 정규화는 원래 R0 유효 TRAIN 후보 5,194개로 다시 계산했다. 지원점이 0인 유효 후보도 삭제하지 않는 구현이며, 이번 실제 TRAIN에는 그런 후보가 없었다.', '',
        f"독립 구현은 전체 descriptor와 지원점, token 해시, 정규화를 확인했다. FP64 표본화 비교의 최대 절대 차이는 {max(z['comparison']['max_absolute'] for z in v['models'].values()):.9g}이며 고정 허용오차 atol=rtol=2e−6을 통과했다. 정규화와 지원점은 정확히 일치했다.", '',
        '생산 직후 저장한 [생성 기록](CONSTRUCTION_KO.md)의 “독립 검산 전” 상태는 당시 기록으로 보존했다. 이후 [독립 검산](INPUT_VERIFICATION_KO.md)이 완료되었으며 현재 상태는 PASS다.', '',
        '## 사용 이미지와 치수', '',
        '아래는 이전 입력 감사와 같은 첫 6개 적격 합성 TRAIN 이미지다. 각 이미지에 W/H/D(cm), 원래 영상 경계, 기존 R0 투영이 표시되어 있다. 이번 학습 결과나 실사 성능 예시가 아니다. 순서를 고정했고 성능으로 고르지 않았다.', '',
        *[f'![합성 TRAIN 입력과 W/H/D {i}](../pallet_pose_dino_input_audit_20261001_v1/figures/synthetic_train_inputs_{i}.jpg)' for i in range(1,4)], '',
        '## 한계와 재현', '',
        '원본 영역의 token도 DINO의 전역 문맥을 통해 반사 패딩의 영향을 받을 수 있다. backbone forward 자체를 별도 구현으로 재실행한 검증은 아니다. 꼭짓점 평균은 순열에 불변이지만 순서 정보도 버린다. 입력 검산은 T/R 개선 증거가 아니다.', '',
        '[설계](DESIGN_KO.md) · [프로토콜](INPUT_PROTOCOL.json) · [생성 영수증](TRAIN_APPEARANCE_INPUTS.json) · [전체 검산](INPUT_VERIFICATION.json) · [렌더링 데이터](REPORT_DATA.json)', '',
        '대용량 token/NPZ/원본 데이터는 로컬에 보존하며 GitHub에는 올리지 않는다. 공개 코드와 해시만으로 원본 데이터 없이 전체 실험을 재실행할 수는 없다.', '',
        '```bash', 'MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m scripts.research.pallet_pose_dino_native_inputs_20261001_v1.report','```','']
    data=dict(complete=True,code=bind(__file__),input_receipt=bind(DOC/'TRAIN_APPEARANCE_INPUTS.json'),
        verification=bind(DOC/'INPUT_VERIFICATION.json'),models=r['models'],figures=[bind(path)],
        inherited_figures=[bind(OLD/'figures'/f'synthetic_train_inputs_{i}.jpg') for i in range(1,4)],
        new_fits=0,new_image_forwards=0,new_metric_calls=0,performance_improvement_measured=False)
    for name,value in [('REPORT_KO.md','\n'.join(lines)),('REPORT_DATA.json',json.dumps(data,ensure_ascii=False,indent=2)+'\n')]:
        with (DOC/name).open('x') as f:f.write(value)
    print('NATIVE_INPUT_REPORT_COMPLETE',bind(DOC/'REPORT_KO.md'),flush=True)

if __name__=='__main__':main()
