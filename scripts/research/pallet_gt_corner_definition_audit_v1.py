"""Read-only annotation/geometry audit. Never estimates a new GT or pose."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scripts.annotate.annotate_pnp import make_pallet_keypoints_3d_diagram, project_3d

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / '_docs/experiments/pallet_gt_corner_definition_audit_v1'
PREV = ROOT / '_docs/experiments/pallet_posefix_crop_completion_v2'
RAW = ROOT / 'data/pallet/results/pallet_posefix_crop_completion_v2'

def read(p):
    return json.loads(p.read_text())

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not DOC.exists(), 'Do not overwrite completed audit'
    records = read(PREV / 'INPUT_LOCK.json')['eval_records']
    before = {r['annotation']['path']: digest(ROOT/r['annotation']['path']) for r in records}
    byid = {r['id']:r for r in records}
    counts = defaultdict(Counter)
    for r in records:
        obj = read(ROOT/r['annotation']['path'])['objects'][0]
        counts[r['kind']]['images'] += 1
        anns = obj.get('keypoint_annotations', [])
        for j in range(8):
            ann = anns[j] if j < len(anns) else {}
            counts[r['kind']]['source_' + str(ann.get('source', 'unknown'))] += 1
        counts[r['kind']]['has_extrapolated_mask'] += obj.get('extrapolated_mask') is not None
    # Fixed examples: first crop-support case, first CAD record, first fixed green case.
    gallery = read(RAW / 'GALLERY_SELECTION.json')
    selected = [(gallery[0]['id'], gallery[0]['corners']),
                (next(r['id'] for r in records if r['session']=='eval_cad'), [0,4,5]),
                (next(g for g in gallery if g['group']=='RANDOM_GREEN_FIXED')['id'], [0,1,4,5])]
    DOC.mkdir()
    (DOC/'figures').mkdir()
    summaries=[]
    for i,(fid, corners) in enumerate(selected,1):
        rec=byid[fid]; annotation=read(ROOT/rec['annotation']['path']);obj=annotation['objects'][0]
        camera=annotation['camera_data'];intr=camera['intrinsics']
        K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
        cf=obj.get('camera_facing_pnp') or {};dims=cf.get('dimensions_m',obj['dimensions_m'])
        pose=np.asarray(cf.get('pose_transform',obj['pose_transform']),float)
        xyz=make_pallet_keypoints_3d_diagram(dims['width'],dims['depth'],dims['height'])
        projected=project_3d(xyz,pose[:3,:3],pose[:3,3],K)
        gt=np.asarray(obj['projected_cuboid'],float)
        manual=obj.get('manual_kps',[]);anns=obj.get('keypoint_annotations',[])
        im=np.asarray(Image.open(ROOT/rec['image']['path']).convert('RGB'))
        assert im.shape[:2]==(camera['height'],camera['width'])
        rows=[]
        for j in corners:
            ann=anns[j] if j<len(anns) else {}
            p=projected[j];valid=not np.array_equal(p,[-1,-1]) and np.isfinite(p).all()
            rows.append(dict(corner=j,source=ann.get('source','unknown'),visibility=ann.get('visibility','unknown'),
                saved_vs_pose_px=float(np.linalg.norm(gt[j]-p)) if valid else None,
                equals_manual_field=bool(j<len(manual) and manual[j] is not None and np.allclose(gt[j],manual[j],atol=1e-8,rtol=0))))
        fig,axs=plt.subplots(1,1+len(corners),figsize=(5*(1+len(corners)),5))
        for k,ax in enumerate(axs):
            ax.imshow(im)
            js=range(8) if k==0 else [corners[k-1]]
            for j in js:
                ax.scatter(*gt[j],s=70,facecolors='none',edgecolors='#ff9800',linewidths=2)
                ax.text(gt[j,0]+3,gt[j,1]-4,f'P{j}',color='#ff9800',fontsize=10)
                if not np.array_equal(projected[j],[-1,-1]):
                    ax.scatter(*projected[j],marker='+',s=85,color='#00cfff')
            if k==0:ax.set_xlim(0,im.shape[1]);ax.set_ylim(im.shape[0],0);ax.set_title('Stored label (orange) / saved-pose projection (cyan)')
            else:
                j=corners[k-1];x,y=gt[j];ax.set_xlim(max(0,x-65),min(im.shape[1],x+65));ax.set_ylim(min(im.shape[0],y+65),max(0,y-65))
                ax.set_title(f'P{j}: source={rows[k-1]["source"]}\npose residual={rows[k-1]["saved_vs_pose_px"]:.2f}px')
        fig.suptitle(f'Example {i} | diagnostic only: cyan is NOT independent truth',fontsize=15)
        fig.tight_layout();fig.savefig(DOC/'figures'/f'example_{i:02d}.png',dpi=110);plt.close(fig)
        summaries.append(dict(example=i,kind=rec['kind'],corners=rows))
    assert all(digest(ROOT/p)==h for p,h in before.items())
    output=dict(images=len(records),counts={k:dict(v) for k,v in counts.items()},examples=summaries,
                all_annotations_unchanged=True,no_training=True,no_new_GT=True,
                projection_is_independent_truth=False,physical_identity_not_resolved=True)
    (DOC/'AUDIT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    lines=['# 3D 코너 정의와 기존 GT 저장 방식 감사','',
        '**현재 GT를 전부 바꾸거나 재어노테이션할 근거는 확인되지 않았다.** 3D는 이상적 직육면체이고, 2D는 기존 클릭을 보존하는 구조다. 둥근 코너에서 두 정의가 달라질 가능성은 있지만 이 감사만으로 실제 정답 오차를 확정하지 않는다.','',
        '## 코드에서 확인한 사실','',
        '- `scripts/annotate/annotate_pnp.py:169`: P0…P7은 (±W/2, ±H/2, ±D/2), P8은 중심이다. 곡률·깎임·표면 세부 형상을 표현하지 않는다.',
        '- `scripts/annotate/annotate_io.py:474`: `make_annotation`은 클릭한 좌표를 그대로 `projected_cuboid`에 저장한다. 이 필드 이름만 보고 모두 PnP 투영이라고 판단하면 안 된다. 미입력 점은 저장된 pose projection으로 채울 수 있다.',
        '- `scripts/annotate/annotate.py:1207`: 기존 어노테이션 도구에 이미 T/TWO-LINE 기능이 있다. 두 선의 각 두 점, 총 네 클릭으로 교차점을 계산하고 source=extrapolated로 표시한다. 새로운 필수 어노테이션 절차가 아니다.',
        '- source=manual_click은 입력 방법이지 실제 외형/가상 직육면체 구분이 아니다. source=unknown과 manual_kps 필드만으로 과거 사람의 의도나 입력 출처를 복원할 수 없다.',
        '- 현재 TWO-LINE 입력 코드는 visibility=1도 함께 설정한다. 따라서 그 값만으로 실제 가려짐을 단정하지 않는다. 이번 감사에서는 수정하지 않았다.','',
        '## 동결 평가 입력의 출처 기록','',
        '아래는 최근 실험 입력 278장에 저장된 8개 코너의 출처 메타데이터 수이다. 평가 유효 마스크 적용 후의 정확도 분모가 아니며 unknown은 정답 오류를 뜻하지 않는다.','',
        '|종류|이미지|코너 출처 기록|','|---|---:|---|']
    for kind,c in counts.items():lines.append(f'|{kind}|{c["images"]}|'+', '.join(f'{k.removeprefix("source_")}={v}' for k,v in c.items() if k.startswith('source_'))+'|')
    lines+=['','## 실제 사진 예시','',
        '주황 원=저장된 2D 좌표, 하늘색 +=저장된 pose와 치수로 다시 투영한 점. **하늘색은 더 정확한 새 정답이 아니다.** 같은 클릭으로 추정한 pose일 수 있으므로 둘의 차이는 적합 잔차일 뿐 GT 정확도나 둥근 모서리 오차가 아니다. 예측 모델은 표시하지 않았다. 모든 그림은 카메라 동적 P 인덱스이며 앞선 대칭 평가의 G 인덱스와 혼동하면 안 된다.']
    for s in summaries:
        lines += ['',f'### 예시 {s["example"]} — {s["kind"]}','',f'![saved label and pose](figures/example_{s["example"]:02d}.png)','',
                  '|점|기록된 출처|저장 pose와 차이(px)|manual_kps 필드와 동일|','|---|---|---:|---|']
        for r in s['corners']:lines.append(f'|P{r["corner"]}|{r["source"]}|{r["saved_vs_pose_px"]:.2f}|{r["equals_manual_field"]}|')
    lines+=['','## 지금 할 일','',
        '사용자의 새 클릭·선 지정·GT 전체 변경은 중단한다. 명확한 코너는 기존 GT 유지, 애매한 코너는 미확인으로 남긴다. 특정 기존 점을 고치려면 그 점과 대응하는 바깥 직선/3D 정의가 독립적으로 확인돼야 한다. PnP 잔차가 작다고 정답임을 보장하거나, 잔차가 크다고 틀렸다고 판정하지 않는다.',
        '',f'읽은 annotation {len(before)}개 SHA256 전후 일치. 학습·pose 재추정·기존 GT/모델/평가 결과 변경 없음.']
    (DOC/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
