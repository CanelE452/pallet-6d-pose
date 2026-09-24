"""Selection reads only the frozen split metadata and raw image pixels."""
from collections import Counter
import hashlib
import subprocess
import cv2
import numpy as np
from . import common as C

def order(fid, seed=20260923):
    return hashlib.sha256(f'{seed}:{fid}'.encode()).hexdigest()

def select(pool, thumbs):
    chosen = []; relaxations = []
    for severity in C.SEVERITIES:
        candidates = [r for r in pool if r['severity'] == severity]
        local = []; counts = Counter(); cap = 2
        while len(local) < 6:
            eligible = [r for r in candidates if r not in chosen and counts[r['recording']] < cap
                        and all(float(np.abs(thumbs[r['frame_id']] - thumbs[q['frame_id']]).mean()) > 2 for q in chosen)]
            if not eligible:
                if cap >= 6:
                    raise RuntimeError('Cannot obtain six non-duplicate images: stop without model-dependent fallback')
                cap += 1
                relaxations.append(dict(severity=severity, cap=cap,
                    reason='Available recording/candidate/near-duplicate constraints exhausted at lower cap',
                    pool_counts=dict(Counter(r['recording'] for r in candidates))))
                continue
            min_count = min(counts[r['recording']] for r in eligible)
            eligible = [r for r in eligible if counts[r['recording']] == min_count]
            def rank(r):
                distance = min((float(np.abs(thumbs[r['frame_id']]-thumbs[q['frame_id']]).mean()) for q in local), default=0.)
                return (-distance, order(r['frame_id']))
            r = min(eligible, key=rank)
            chosen.append(r); local.append(r); counts[r['recording']] += 1
    assert len({r['recording'] for r in chosen}) >= 3
    return chosen, relaxations

def guide(contract):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz = np.array(contract['xyz_m'])
    # Fixed pedagogical projection of the actual canonical coordinates, never an image pose.
    uv = np.column_stack((xyz[:,0] + .48*xyz[:,2], -xyz[:,1] + .35*xyz[:,2]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for a,b in contract['edges']:
        ax.plot(uv[[a,b],0], uv[[a,b],1], color='#777777', lw=2)
    for i,(x,y) in enumerate(uv):
        ax.scatter(x,y,c='#dc6d21' if i<4 else '#1687bd',s=70)
        ax.annotate(f'P{i}',(x,y),xytext=(4,8 if i in (0,1,4,5) else -16),textcoords='offset points',fontsize=14)
    ax.set_title('Camera-facing corner index (schematic)\nNear: P0-3 / Far: P4-7 / Top: P0,1,4,5')
    ax.axis('off'); ax.margins(.15); fig.tight_layout()
    fig.savefig(C.DOC/'figures/corner_index_reference.png',dpi=150); plt.close(fig)
    rows='\n'.join(f'|P{i}|{n}|{contract["xyz_m"][i]}|' for i,n in enumerate(contract['names']))
    C.save_new(C.DOC/'ANNOTATION_GUIDE_KO.md', f'''# 모델을 보지 않는 코너 확인 안내

![번호 안내](figures/corner_index_reference.png)

## 어디를 확인하나요?

기존 camera-facing 번호를 그대로 사용합니다. 가까운 면이 P0~P3, 반대편이 P4~P7입니다.
윗면은 P0·P1·P5·P4, 아랫면은 P3·P2·P6·P7입니다. P8 중심은 찍지 않습니다.
45도 등에서 가까운 면/번호를 확실히 정할 수 없으면 **UNCERTAIN**으로 두세요.
그림은 번호 안내일 뿐, 실제 이미지의 자세나 정답을 제시하지 않습니다.

물리 치수 W/D/H = 1.10/1.30/0.11 m, 물리 XYZ = 1.10/0.11/1.30 m.
시점에 따라 카메라를 향한 면의 W/D 배치가 달라질 수 있습니다. 항상 짧은 변이 정면이라는 뜻이 아닙니다.
둥근 외형의 끝과 이상적인 직육면체 꼭짓점은 다를 수 있습니다.
직선 연장으로만 추정할 수 있으면 VIRTUAL_INFERABLE이며 주평가에서 제외합니다.

|번호|기존 이름|local XYZ (m)|
|---|---|---|
{rows}

edge graph: `{contract['edges']}`

## 조작

1. 이름은 선택 입력입니다. 비워 두면 익명 로컬 검수자로 기록됩니다. P0~P7 버튼 또는 숫자 0~7로 점을 선택합니다.
2. 미확인 점을 클릭하면 ‘직접 보임’으로 저장하고 다음 번호로 자동 이동합니다. 직선 연장 추정점은 먼저 V를 누르고 클릭하세요.
3. S: 자체 가림, E: 외부 물체 가림, O: 화면 밖, U: 판단 불가 → 클릭 없이 상태를 저장하고 다음 번호로 이동합니다. P7에서는 멈춥니다.
4. 마우스 휠 확대/축소, 오른쪽 버튼 드래그 이동, F 전체 보기.
5. Ctrl+Z 취소. 이전/다음 버튼으로 이동. 변경은 자동 저장됩니다.
6. 18장 각각 8개 상태를 정하면 ‘완료 확인’을 누릅니다.

아무 상태도 선택하지 않은 빈칸은 UNCERTAIN과 다릅니다. 8개 모두 직접 확인하세요.
DIRECT_VISIBLE만 fixed-identity 주평가에 쓰고 가려진 점을 PnP로 채우지 않습니다.
첫 화면에는 예측, 기존 GT, confidence, pose가 전혀 없습니다.

출처: `scripts/annotate/annotate_draw.py`의 KP_NAMES/CUBOID_EDGES,
`scripts/annotate/annotate_pnp.py`의 PALLET_DIMS/make_pallet_keypoints_3d_diagram.
''')

def main():
    assert not C.DOC.exists() and not C.RAW.exists(), 'Existing namespace: do not rerun selection'
    source = C.read(C.SPLIT)
    assert len(source['heldout']) == 128
    # Explicit whitelist: never load legacy coordinates, confidence, predictions or errors.
    pool = [dict(frame_id=r['id'], severity=r['severity'], recording=r['recording_group'], image=r['image']) for r in source['heldout']]
    assert not {r['recording'] for r in pool} & {'REC_001','REC_002'}
    thumbs={}
    for r in pool:
        path=C.ROOT/r['image']['path']; assert C.sha(path)==r['image']['sha256']
        im=cv2.imread(str(path)); assert im is not None
        r['hw']=list(im.shape[:2])
        thumbs[r['frame_id']]=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,48)).astype(np.float32)
    frames,relax=select(pool,thumbs)
    for r in frames:
        r['selection_reason']='severity6; recording-balanced; thumbnail L1 farthest-point; seeded hash ties; MAD>2'
        r['no-model-output-used']=True
    selection=dict(seed=20260923,frames=frames,relaxations=relax,selected_utc=C.now(),
                   source_split_sha256=C.sha(C.SPLIT),near_duplicate_rule='64x48 grayscale MAD <=2 on 0..255',
                   model_output_used=False,private_frame_mapping=True)
    C.RAW.mkdir(parents=True); (C.DOC/'figures').mkdir(parents=True); C.OUT.mkdir(parents=True)
    C.save_new(C.RAW/'ANCHOR_SELECTION.json',selection)
    C.save_new(C.RAW/'WORKTREE_BEFORE.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          status=subprocess.check_output(['git','status','--short'],text=True)))
    contract=C.contract(); C.save_new(C.DOC/'CORNER_CONTRACT.json',contract); guide(contract)
    template=dict(schema='verified_anchor_v1',annotator='',selection_sha256=C.sha(C.RAW/'ANCHOR_SELECTION.json'),
        frames=[dict(frame_id=r['frame_id'],image_sha256=r['image']['sha256'],annotator='',
                     **{'pass':1},corners=[dict(id=i,status=None,xy=None,note=None) for i in range(8)]) for r in frames])
    C.save_new(C.RAW/'LABEL_TEMPLATE.json',template)
    C.save_new(C.RAW/'ANNOTATION_PROGRESS.json',C.progress(template,selection))
    C.save_new(C.DOC/'INPUT_BINDINGS.json',dict(split=dict(path=str(C.SPLIT.relative_to(C.ROOT)),sha256=C.sha(C.SPLIT)),
        selection_sha256=C.sha(C.RAW/'ANCHOR_SELECTION.json'),contract_sha256=C.sha(C.DOC/'CORNER_CONTRACT.json'),
        selection_location='local/private data/pallet/results/pallet_verified_anchor_v1/ANCHOR_SELECTION.json'))
    C.save_new(C.DOC/'ANCHOR_SELECTION_SUMMARY.json',dict(n=18,pool=128,seed=20260923,
        severity=dict(Counter(r['severity'] for r in frames)),recordings=len({r['recording'] for r in frames}),
        recording_severity=[dict(severity=s,counts=sorted(Counter(r['recording'] for r in frames if r['severity']==s).values())) for s in C.SEVERITIES],
        relaxations=relax,private_full_mapping_not_pushed=True))
    C.save_new(C.DOC/'PREFLIGHT_AUDIT.md', '# Preflight\n\n'+
        f'HEAD: {subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}\n\n'+
        'Frozen HELDOUT128 only; REC_001/REC_002 excluded. Selection reads whitelisted split fields and RGB only. '+
        '18 images, 6 per severity. No training, optimizer, inference, pseudo generation, GT writes or model selection. '+
        'Clean recording cap relaxed only if candidates/near-duplicate constraints require it; details in selection summary. '+
        'Coordinates and original images stay local. Source split and canonical source hashes are locked.\n')
    C.save_new(C.DOC/'REPORT_KO.md','''# Minimal verified visible-corner anchor — 준비 보고서

## 현재 상태

**WAITING_FOR_HUMAN_LABELING · 0/18장**. 새 학습·추론·평가 0회.
기존 HELDOUT128의 플라스틱에서 CLEAN/MODERATE/SEVERE 각 6장을 모델 결과 없이 선정했습니다.
기존 GT와 모델 성능은 아직 비교하지 않았습니다. 결과 수치는 어노테이션 완료 전에는 만들지 않습니다.

## 점 정의와 확인 방법

![기존 저장소 코너 정의](figures/corner_index_reference.png)

[조작 안내](ANNOTATION_GUIDE_KO.md)를 따라 각 장 P0~P7의 상태를 확인합니다.
직접 확인 가능한 점만 클릭하고, 가림·화면 밖·판단 불가인 점은 상태만 선택합니다.
18장 × 8점 = 144개 상태이며, 144번 클릭하라는 뜻이 아닙니다.
현재 확인 이미지/직접 클릭/QA 재검토 모두 0. 시간은 아직 측정하지 않았습니다.

## 범위 및 선정

녹색/목재는 이 파일럿 대상이 아닙니다. REC_001/REC_002는 제외했습니다.
난도별 기록 균형을 우선하고, 64×48 grayscale 이미지 간 거리로 다양성을 확보했습니다.
MAD≤2/255 근접 중복은 제외. 고정 seed 20260923. 모델 오차·confidence·기존 좌표를 선정에 쓰지 않았습니다.
정확한 매핑/원본/좌표는 local private 경로에 두고 공개하지 않습니다.
[선정 집계](ANCHOR_SELECTION_SUMMARY.json)에 recording 제한 완화와 이유를 기록했습니다.

## 이후 절차

사람의 첫 입력을 잠근 뒤 coverage 확인 → 필요할 때만 최대 6장 추가 → 일부 점 QA → frozen 예측 재채점.
60개 DIRECT_VISIBLE, 난도별 12개, 5개 이상의 ID 각각 3회, 최소 3 recording이 최초 coverage 기준입니다.
18장으로 충분하면 추가 0장, 최대 24장 이후 추가 요구하지 않습니다.
이번 단계는 UI/선정 준비까지이며 후속 QA/재평가를 완료했다고 주장하지 않습니다.
새 좌표는 TRAIN에 쓰지 않으며 기존 legacy GT를 수정하지 않습니다.
균형 소규모 reused-DEV 진단이며 독립 TEST/운영 평균/hidden GT/6D GT가 아닙니다.

## 재현

`python -m scripts.research.pallet_verified_anchor_v1.label_anchor`

입력 잠금은 INPUT_BINDINGS.json, 코너 계약은 CORNER_CONTRACT.json을 참조하세요.
실제 이미지 예시는 blind first pass 완료 전 보고서에 생성하지 않습니다.
''')
    print('WAITING_FOR_HUMAN_LABELING 0/18',relax,flush=True)

if __name__=='__main__': main()
