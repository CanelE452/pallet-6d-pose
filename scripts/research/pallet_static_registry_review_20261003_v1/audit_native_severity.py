"""Independently verify actual native severity records without changing labels or GT."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT/'data/pallet/results/pallet_static_registry_review_20261003_v1/native_severity_review'
MANIFEST = ROOT/'_docs/experiments/pallet_static_registry_review_20261003_v1/review/STATIC_REVIEW_MANIFEST.json'
OUT = ROOT/'_docs/experiments/pallet_static_registry_review_20261003_v1/native_severity_check_20261004_v1'
LEVEL = {'clean':'none', 'moderate':'partial', 'severe':'heavy'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    store_path=INPUT/'STATIC_SEVERITY_INPUTS.json'
    group_path=INPUT/'STATIC_SEVERITY_GROUPS.json'
    store_bytes=store_path.read_bytes()
    store=json.loads(store_bytes)
    groups=json.loads(group_path.read_text())
    manifest=json.loads(MANIFEST.read_text())
    cases={case['case_id']:case for case in manifest['cases']}
    records=store['records']
    checks={}
    checks['unique_frozen_case_ids']=len(cases)==len(manifest['cases'])==438
    checks['manifest_binding']=store['input_bindings']['static_manifest_sha256']==digest(MANIFEST)
    checks['source_bindings']=store['input_bindings']['source_bindings']==manifest['source_bindings'] and all(
        digest(ROOT/b['path'])==b['sha256'] and (ROOT/b['path']).stat().st_size==b['bytes'] for b in manifest['source_bindings'])
    checks['raw_images_preserved']=all(digest(ROOT/c['image']['path'])==c['image']['sha256']
        and (ROOT/c['image']['path']).stat().st_size==c['image']['bytes'] for c in cases.values())
    checks['original_annotations_preserved']=all(digest(ROOT/c['annotation']['path'])==c['annotation']['sha256']
        and (ROOT/c['annotation']['path']).stat().st_size==c['annotation']['bytes'] for c in cases.values())
    checks['record_ids']=set(records).issubset(cases) and all(fid==r['frame_id'] for fid,r in records.items())
    checks['record_image_and_population']=all(r['image_sha256']==cases[fid]['image']['sha256']
        and r['dataset_population']==cases[fid]['population'] and r['session_id']==cases[fid]['session'] for fid,r in records.items())
    checks['label_mapping']=all(r['severity'] in LEVEL and r['occlusion_level']==LEVEL[r['severity']] for r in records.values())
    checks['preserved_prior_label_provenance']=all(r['prior_approved_severity']==cases[fid]['frame_severity'] for fid,r in records.items())
    checks['honest_input_provenance']=all(r['label_source']=='HUMAN_DIRECT_CLASS'
        and r['input_action'] in ('human_keyboard','human_mouse_button')
        and r['actor_id']==store['actor']['id'] and r['human_identity_confirmed'] is False for r in records.values())
    checks['real_selection_times']=all(datetime.fromisoformat(r['selected_at'].replace('Z','+00:00'))>=
        datetime.fromisoformat(r['started_at'].replace('Z','+00:00')) and r['elapsed_view_seconds']>=0 for r in records.values())
    checks['no_corner_or_pose_promotion']=all(r['corner_review_status']=='NOT_REVIEWED'
        and r['target_identity_status']=='NOT_CONFIRMED' and r['pose_reference_status']=='NOT_REVIEWED' for r in records.values())
    replay={}
    for event in store['history']:
        if event['action']=='explicit_class_selection':replay[event['frame_id']]=event['current']
        elif event['action']=='human_reset_to_unreviewed':replay.pop(event['frame_id'],None)
        else:raise ValueError('Unknown history action '+event['action'])
    checks['history_replay_equals_latest']=replay==records
    local={p.stem:json.loads(p.read_text())['record'] for p in (INPUT/'severity_annotations').rglob('*.json')}
    checks['individual_json_equals_latest']=len(local)==len(records) and all(local[Path(cases[fid]['image']['path']).stem]==r for fid,r in records.items())
    effective=[];changed=[];reused=[]
    for fid,c in cases.items():
        current=records.get(fid)
        severity=current['severity'] if current else c['frame_severity']['status'] if c['frame_severity']['locked'] else 'unreviewed'
        source='new_explicit_human_input' if current else 'existing_approved_label' if c['frame_severity']['locked'] else 'unreviewed'
        row=dict(case_id=fid,frame_id=c['frame_id'],population=c['population'],session=c['session'],severity=severity,
            source=source,image=c['image'],original_approved=c['frame_severity'])
        effective.append(row)
        if current and c['frame_severity']['locked'] and c['frame_severity']['status']!=severity:
            changed.append(dict(case_id=fid,before=c['frame_severity']['status'],after=severity))
        if not current and c['frame_severity']['locked']:reused.append(row)
    expected={(r['population'],r['severity'],r['case_id']) for r in effective}
    actual=[(population,severity,row['frame_id']) for population,labels in groups['groups'].items()
            for severity,rows in labels.items() for row in rows]
    checks['groups_exact_and_unique']=len(actual)==len(set(actual))==438 and set(actual)==expected
    counts={population:dict(Counter(r['severity'] for r in effective if r['population']==population))
        for population in ('DEV319','GREEN0918')}
    checks['all_frame_severities_available']=all(r['severity'] in LEVEL for r in effective)
    checks['inputs_unchanged_during_audit']=store_bytes==store_path.read_bytes()
    receipt=dict(status='PASS' if all(checks.values()) else 'FAIL',checked_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        checks=checks,counts=counts,new_records=len(records),history_events=len(store['history']),
        changed_rectangular=len(changed),reused_existing=reused,source_sha256=digest(store_path),
        manifest_sha256=digest(MANIFEST),actor=store['actor'],
        scope='frame severity record and immutable input verification; no class relabeling, new inference or training',
        paper_insertion_status='NOT_INSERTED',downstream_status='native inputs not yet consumed by existing metrics/manuscript pipeline',
        classification_semantics_status='AWAITING_USER_CLARIFICATION',
        visual_check=dict(status='LABELS_PRESERVED_SEMANTICS_NEED_CONFIRMATION',
            examples=['DEV319::eval_cad:1778653017736058368','DEV319::eval_cad:1778653018878592768','DEV319::eval_cad:1778653041187027200'],
            observation='Displayed moderate rectangular examples appear to lack an externally blocking object; confirm occlusion-only versus overall difficulty before a paper claim.'))
    (OUT/'STATIC_SEVERITY_INPUTS_SNAPSHOT.json').write_bytes(store_bytes)
    write(OUT/'VALIDATION.json',receipt)
    write(OUT/'RECTANGULAR_LABEL_CHANGES.json',changed)
    write(OUT/'EFFECTIVE_FRAME_LABELS.json',dict(schema='validated_static_native_effective_frame_labels_v1',
        source_sha256=receipt['source_sha256'],manifest_sha256=receipt['manifest_sha256'],rows=effective,
        human_identity_confirmed=False,corner_visibility_status='NOT_NEWLY_REVIEWED',paper_insertion_status='NOT_INSERTED'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,axes=plt.subplots(6,3,figsize=(12,17),squeeze=False)
    for i,(population,label) in enumerate((p,l) for p in ('DEV319','GREEN0918') for l in LEVEL):
        subset=[r for r in effective if r['population']==population and r['severity']==label]
        for j,ax in enumerate(axes[i]):
            ax.axis('off')
            if j<len(subset):
                row=subset[j]
                ax.imshow(Image.open(ROOT/row['image']['path']))
                ax.set_title(f'{population} / {label}\n{row["frame_id"]}',fontsize=9)
        axes[i,0].set_ylabel(f'{population} {label}',fontsize=10)
    fig.suptitle('Saved human labels: first 3 frames in frozen order per group',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'SAVED_LABEL_EXAMPLES.png',dpi=130);plt.close(fig)
    lines=['# 직사각형·정사각형 가림 분류 저장 검증','',
        f'검증 결과: **{receipt["status"]} · {sum(checks.values())}/{len(checks)} 검사 통과**. 실제 저장 파일과 수정 이력을 원본 평가 이미지에 대조했다. 가림 등급을 자동으로 바꾸지 않았다.','',
        '| 평가 자료 | 가림 없음 | 중간 | 어려움 | 미분류 |','|---|---:|---:|---:|---:|']
    for population,name in [('DEV319','직사각형 319장'),('GREEN0918','별도 촬영 정사각형 119장')]:
        count=counts[population];lines.append(f'| {name} | {count.get("clean",0)} | {count.get("moderate",0)} | {count.get("severe",0)} | {count.get("unreviewed",0)} |')
    lines += ['',f'이번 신규 입력은 {len(records)}장(직사각형 316장, 정사각형 119장)이다. 직사각형 3장은 기존 승인 분류를 재사용했다. 이 중 한 장의 R 초기화 2회는 이력에 남아 있으며 기존 분류로 돌아갔다.','',
        f'직사각형의 이전 151/87/81 분류에서 {len(changed)}장 등급이 변경됐다. 원래 승인 결과·manifest·원본 이미지·GT는 보존했다. 새 판정은 별도 수정본에 저장됐다.', '',
        '재사용한 3장:']
    lines += [f'- `{r["frame_id"]}`: `{r["severity"]}`' for r in reused]
    lines += ['',f'원본 438장 이미지와 GT·출처 해시, 개별 {len(records)}개 저장 JSON, {len(store["history"])}건의 이력 재생, 집단 목록의 분모와 중복 여부를 확인했다.', '',
        '가림 등급 입력은 완료됐다. 코너 가시성·대상 대응·독립 6D 참조는 이번 입력으로 검수 완료 처리하지 않았다. 식별자는 자동 로컬 별칭이며 신원·예측 노출 이력 확인 상태도 그대로 보존했다.', '',
        '**이미지 내용 검수에서 분류 기준 확인이 남았다.** 아래 직사각형 중간 예시 3장에서는 외부 물체의 가림이 보이지 않는 것으로 판단된다. 이는 원본을 본 질적 관찰이며 사용자의 등급을 자동으로 변경하지 않았다. 실제 외부 가림 정도인지 시점·거리·조명 등을 포함한 전체 난이도인지 사용자에게 확인을 요청했다. 저장 무결성 PASS를 가림 판정 정답성 PASS로 해석하면 안 된다.','',
        '**원고·실험 지표 반영은 아직 하지 않았다.** 기존 보고서는 이전 직사각형 집계와 정사각형 대기 상태를 사용한다. 분류 기준을 확인한 뒤 새 검증 집계를 별도 수정본으로 연결해야 하며, 아래 검증 파일이 그 입력을 고정한다.','',
        '[검산 결과](VALIDATION.json) · [42장 변경 목록](RECTANGULAR_LABEL_CHANGES.json) · [유효한 438장 분류와 출처](EFFECTIVE_FRAME_LABELS.json)','',
        '아래 이미지는 각 집단에서 고정 순서의 첫 3장을 보여준다. 모델 예측·오차를 표시하지 않았다. 표본 그림만으로 전체 가림 판정의 정답성을 보증하지 않는다.','',
        '![저장된 등급별 원본 예시](SAVED_LABEL_EXAMPLES.png)','',
        '실행 명령:','',
        '```bash','/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_static_registry_review_20261003_v1/audit_native_severity.py','```','',
        '새 학습·optimizer update·추론·원본 덮어쓰기·외부 업로드·push·PDF 생성: 0회.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','counts','new_records','history_events','changed_rectangular')},ensure_ascii=False))
    print('REPORT',OUT/'REPORT_KO.md')
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
