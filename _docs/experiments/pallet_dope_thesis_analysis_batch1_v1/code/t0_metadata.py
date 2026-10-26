"""T0: 메타데이터만 집계한다. RGB/원본 annotation/체크포인트를 열지 않는다."""
from common import *
from collections import Counter, defaultdict
import argparse

WORK = REPO/'data/evaluation/pallet_eval_v1'
REGISTRY = REPO/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
MISSING = '없음'
LABELS = {'PLASTIC':'직사각형 플라스틱','WOOD':'소형 목재','GREEN':'녹색 플라스틱',
          'NONE':'파렛트 없음','UNKNOWN':'종류 미상'}
POP_LABELS = {'DEV292':'주 평가 DEV292','PLASTIC184':'플라스틱 184 보조 평가',
    'TRAIN_PLASTIC':'플라스틱 학습 후보','TRAIN_WOOD':'목재 학습 후보','TRAIN_GREEN':'녹색 학습 후보',
    'PHYSICAL_FINAL_POSITIVE':'물리 FINAL 양성','PHYSICAL_FINAL_NEGATIVE':'물리 FINAL 음성'}


def present(value):
    return value is not None and str(value).strip().lower() not in ('','unknown','none','null','n/a','na','없음','미상')


def metadata_path(value, base=REPO):
    p = Path(value)
    return p if p.is_absolute() else base/p


def repo_key(value):
    p = metadata_path(value)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


class Metadata:
    def __init__(self):
        self.files = {}
        self.cache = {}

    def allowed(self, path):
        path = Path(path)
        if 'annotations' in path.parts or path.suffix.lower() not in ('.json','.csv','.md'):
            raise RuntimeError('메타데이터 외 본체 접근 금지: '+str(path))
        if path.name.endswith('.pth') or path.name.startswith('PREDICTIONS') or path.name.startswith('SCORES'):
            raise RuntimeError('T0에서 예측/점수/모델 읽기 금지: '+str(path))
        return path

    def read(self, path, kind='json'):
        path = self.allowed(path)
        if str(path) in self.cache:
            return self.cache[str(path)]
        digest = sha(path)
        if kind == 'csv':
            with path.open() as stream:
                value = list(csv.DictReader(stream))
        elif kind == 'text':
            value = path.read_text()
        else:
            value = read(path)
        if sha(path) != digest:
            raise RuntimeError('읽는 동안 메타데이터가 바뀜: '+str(path))
        self.files[str(path)] = {'path':str(path),'sha256':digest,'종류':kind}
        self.cache[str(path)] = value
        return value

    def verify(self):
        for path, entry in self.files.items():
            if sha(path) != entry['sha256']:
                raise RuntimeError('집계 중 메타데이터가 바뀜: '+path)


def nested_sources(value, source):
    yield value, source
    for key in ('conditions','default_tags','defaults','acquisition','reference_metadata'):
        if isinstance(value.get(key),dict):
            yield value[key],source+'/'+key


def pick_condition(sources, name):
    keys = {'lighting':('lighting',), 'acquisition_domain':('acquisition_domain',),
        'indoor_outdoor':('indoor_outdoor','environment','environment_type','scene_environment','location_type')}[name]
    for obj, path in sources:
        for key in keys:
            value = obj.get(key)
            if not present(value) or isinstance(value,(dict,list)):
                continue
            if name == 'lighting':
                value = {'day':'day','daylight':'day','주간':'day','night':'night','야간':'night'}.get(str(value).lower())
            elif name == 'indoor_outdoor':
                value = {'indoor':'indoor','실내':'indoor','outdoor':'outdoor','실외':'outdoor',
                    'mixed':'mixed','both':'mixed','실내외':'mixed'}.get(str(value).lower())
            else:
                value = str(value)
            if value is not None:
                return value,path+'/'+key
    return None,None


def truth_claims(sources):
    result = []
    keys = ('independent_6d_ground_truth','independent_pose_ground_truth','external_pose_ground_truth',
            'pose_ground_truth_source','ground_truth_pose_source','pose_reference_source','pose_measurement_method')
    for obj, path in sources:
        for key in keys:
            if key in obj and present(obj[key]):
                result.append({'값':obj[key],'출처':path+'/'+key})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='root GO 및 before snapshot 완료 후에만 지정')
    args = parser.parse_args()
    if not args.execute:
        parser.error('GO 후 --execute를 지정해야 한다. 기본 동작에서는 집계하지 않는다.')
    before = OUT/'hashes/SNAPSHOT_BEFORE.json'
    if not before.is_file():
        raise RuntimeError('집계 전 원본 before 해시 스냅샷이 필요하다.')
    required = [FLIP/'MANIFEST.json',NOFLIP/'PLASTIC184_MANIFEST.json',
        *[FLIP/k/'POOL.json' for k in KINDS],WORK/'manifests/frames.csv',
        WORK/'manifests/sessions.csv',WORK/'manifests/FINAL_POSITIVE.csv',
        WORK/'manifests/FINAL_NEGATIVE.csv',WORK/'DATASET_CONTRACT.json',WORK/'README.md',REGISTRY]
    start('T0_메타데이터_집계',inputs=required)
    meta = Metadata()
    manifest = meta.read(FLIP/'MANIFEST.json')['records']
    auxiliary = meta.read(NOFLIP/'PLASTIC184_MANIFEST.json')['records']
    pools = {k:meta.read(FLIP/k/'POOL.json')['records'] for k in KINDS}
    frames = meta.read(WORK/'manifests/frames.csv','csv')
    sessions = meta.read(WORK/'manifests/sessions.csv','csv')
    final_manifest = {p:meta.read(WORK/'manifests'/f'FINAL_{p}.csv','csv') for p in ('POSITIVE','NEGATIVE')}
    contract = meta.read(WORK/'DATASET_CONTRACT.json')
    meta.read(WORK/'README.md','text')
    registry = meta.read(REGISTRY)
    assert len(manifest)==len({r['id'] for r in manifest})==292
    assert len(auxiliary)==len({r['id'] for r in auxiliary})==184
    assert all(len(rows)==len({r['id'] for r in rows})==1000 for rows in pools.values())
    by_frame = {r['frame_id']:r for r in frames}
    by_sha = defaultdict(list)
    for row in frames:
        by_sha[row['image_sha256']].append(row)
    by_session = defaultdict(list)
    for index,row in enumerate(sessions):
        by_session[row['session_id']].append((row,f'{WORK}/manifests/sessions.csv#row={index+2}'))
    aliases = defaultdict(list)
    for gi,group in enumerate(registry['groups']):
        if group.get('is_collection'):
            continue
        for si,session in enumerate(group['sessions']):
            if session.get('is_collection'):
                continue
            aliases[session['session_key']].append((group['recording_id'],
                f'{REGISTRY}#/groups/{gi}/recording_id; session=/groups/{gi}/sessions/{si}/session_key'))
    frame_records,populations = {},{}
    session_metadata_missing = set()

    def describe(record, population, source, index):
        fid = record.get('id') or record['session_id']+':'+Path(record['image_path']).stem
        session = record.get('session') or record.get('session_id')
        role = record.get('population_role',record.get('role'))
        image_sha = record.get('image_sha256')
        role_rows = []
        source_frame = record.get('source_frame_id',record.get('frame_id'))
        if source_frame in by_frame:
            role_rows = [by_frame[source_frame]]
        elif image_sha:
            role_rows = by_sha.get(image_sha,[])
        candidates = [(record,f'{source}#/records/{index}')]
        for row in role_rows:
            candidates.append((row,f'{WORK}/manifests/frames.csv#frame_id={row["frame_id"]}'))
        session_entries = []
        for sr,sp in by_session.get(session,[]):
            if role and sr.get('population_role') != role:
                continue
            session_entries.append((sr,sp))
        candidates += session_entries
        session_keys = []
        if session and '/' in session:
            session_keys.append(repo_key(session))
        for sr,_ in session_entries:
            if sr.get('session_path'):
                session_keys.append(repo_key(WORK/sr['session_path']))
        # Path strings identify an explicitly registered folder; no image is opened.
        for row in [record,*role_rows]:
            for key in ('image_path','source_image_path'):
                if row.get(key):
                    base = WORK if row in role_rows and key=='image_path' else REPO
                    p = metadata_path(row[key],base)
                    folder = p.parent.parent if p.parent.name=='rgb' else p.parent
                    session_keys.append(repo_key(folder))
        session_paths = set()
        for sr,_ in session_entries:
            if sr.get('session_metadata_path'):
                session_paths.add(str(metadata_path(sr['session_metadata_path'],WORK)))
        if record.get('camera_source_path') and Path(record['camera_source_path']).name=='session.json':
            session_paths.add(str(metadata_path(record['camera_source_path'])))
        for key in session_keys:
            session_paths.add(str(metadata_path(key)/'session.json'))
        session_docs = []
        for p in sorted(session_paths):
            path = Path(p)
            if 'annotations' in path.parts:
                continue
            if not path.is_file():
                session_metadata_missing.add(str(path))
                continue
            value = meta.read(path)
            if 'objects' in value:
                raise RuntimeError('session.json 대신 annotation 본체가 연결됨: '+str(path))
            session_docs.append((value,str(path)+'#'))
        expanded = [(obj,path) for obj,source_path in [*candidates,*session_docs]
                    for obj,path in nested_sources(obj,source_path)]
        rec_candidates = []
        for obj,path in expanded:
            for key in ('recording_id','source_recording','capture_recording_id','recording','recording_group'):
                if present(obj.get(key)) and isinstance(obj[key],str):
                    rec_candidates.append((obj[key],path+'/'+key))
        for key in dict.fromkeys(session_keys):
            rec_candidates += aliases.get(key,[])
        ids = {value for value,_ in rec_candidates}
        # 기존 registry의 여러 명시 경로가 충돌하면 하나를 추정 선택하지 않는다.
        conflict = len(ids)>1
        recording_id = rec_candidates[0][0] if rec_candidates and not conflict else None
        recording_source = rec_candidates[0][1] if rec_candidates and not conflict else None
        lighting,ls = pick_condition(expanded,'lighting')
        environment,es = pick_condition(expanded,'indoor_outdoor')
        domain,ds = pick_condition(expanded,'acquisition_domain')
        kind = TYPE_TO_KIND.get(record.get('object_type'),record.get('kind'))
        if kind not in KINDS:
            kind = 'NONE' if record.get('object_type')=='none' else 'UNKNOWN'
        claims = truth_claims(expanded)
        item = dict(id=fid,source_frame_id=source_frame,kind=kind,session=session,
            recording_id=recording_id,recording_source=recording_source,
            recording_status='ambiguous_conflicting_metadata' if conflict else ('known_metadata' if recording_id else 'missing_metadata'),
            recording_candidates=sorted(ids),
            recording_evidence=[dict(recording_id=v,source=s) for v,s in rec_candidates],
            recording_resolution='기존 명시 ID 충돌·미확정' if conflict else ('명시 근거 확인' if recording_id else '명시 근거 없음'),
            lighting=lighting,indoor_outdoor=environment,acquisition_domain=domain,
            condition_sources=dict(lighting=ls,indoor_outdoor=es,acquisition_domain=ds),
            image_sha256=image_sha,population_role=role,populations=[population],
            manifest_has_K=('K' in record or isinstance(record.get('camera'),dict) and 'K' in record['camera']),
            independent_6d_truth_metadata_evidence=claims,
            independent_6d_truth_status='메타데이터 명시 있음·정답 본체 미검증' if claims else None,
            source_record=source+'#'+str(index))
        if fid in frame_records:
            prior = frame_records[fid]
            for key in ('image_sha256','kind','session','recording_id','lighting','indoor_outdoor','acquisition_domain'):
                if prior[key] is not None and item[key] is not None and prior[key]!=item[key]:
                    raise RuntimeError('모집단 사이 메타데이터 충돌: '+str((fid,key,prior[key],item[key])))
                if prior[key] is None and item[key] is not None:
                    prior[key]=item[key]
                    if key=='recording_id':
                        prior['recording_source']=item['recording_source']
            if population not in prior['populations']:
                prior['populations'].append(population)
        else:
            frame_records[fid]=item
        return fid

    groups = [('DEV292',manifest,str(FLIP/'MANIFEST.json')),
              ('PLASTIC184',auxiliary,str(NOFLIP/'PLASTIC184_MANIFEST.json'))]
    groups += [('TRAIN_'+kind,rows,str(FLIP/kind/'POOL.json')) for kind,rows in pools.items()]
    physical = [r for r in frames if r['population_role']=='FINAL']
    groups += [('PHYSICAL_FINAL_POSITIVE',[r for r in physical if r['is_positive']=='true'],str(WORK/'manifests/frames.csv')),
               ('PHYSICAL_FINAL_NEGATIVE',[r for r in physical if r['is_positive']=='false'],str(WORK/'manifests/frames.csv'))]
    for population,rows,source in groups:
        populations[population]=[describe(row,population,source,index) for index,row in enumerate(rows)]
        assert len(populations[population])==len(set(populations[population])),population
    assert len(populations['PHYSICAL_FINAL_POSITIVE'])==229,'물리 FINAL 양성229 전제가 달라짐'
    core_hashes = {frame_records[fid]['image_sha256'] for fid in populations['DEV292']}
    positive_hashes = {frame_records[fid]['image_sha256'] for fid in populations['PHYSICAL_FINAL_POSITIVE']}
    overview,recordings,session_counts,conditions,overlaps = [],[],[],[],[]
    for population,ids in populations.items():
        by_kind = defaultdict(list)
        for fid in ids:
            by_kind[frame_records[fid]['kind']].append(frame_records[fid])
        if not ids:
            overview.append({'모집단':POP_LABELS[population],'종류':LABELS['NONE'],
                '영상 수':0,'고유 SHA 수':0,'확인된 녹화 수':0,'녹화 ID 없음':0,
                '근거 없음':0,'메타데이터 상충':0})
        for kind,rows in sorted(by_kind.items()):
            base={'모집단':POP_LABELS[population],'종류':LABELS[kind]}
            known={r['recording_id'] for r in rows if r['recording_id'] is not None}
            overview.append(base|{'영상 수':len(rows),'고유 SHA 수':len({r['image_sha256'] for r in rows}),
                '확인된 녹화 수':len(known),'녹화 ID 없음':sum(r['recording_id'] is None for r in rows),
                '근거 없음':sum(r['recording_status']=='missing_metadata' for r in rows),
                '메타데이터 상충':sum(r['recording_status']=='ambiguous_conflicting_metadata' for r in rows)})
            for rid,n in sorted(Counter(r['recording_id'] for r in rows).items(),key=lambda x:str(x[0])):
                if rid is not None:
                    recordings.append(base|{'녹화 ID':rid,'식별 상태':'명시 근거 확인','영상 수':n})
                else:
                    for status,m in sorted(Counter(r['recording_status'] for r in rows if r['recording_id'] is None).items()):
                        recordings.append(base|{'녹화 ID':MISSING,'식별 상태':'식별 불가(메타데이터 상충)' if status=='ambiguous_conflicting_metadata' else '근거 없음','영상 수':m})
            for sid,n in sorted(Counter(r['session'] for r in rows).items(),key=lambda x:str(x[0])):
                session_counts.append(base|{'세션':sid or MISSING,'영상 수':n})
            for key,label in [('lighting','주간·야간'),('indoor_outdoor','실내·실외'),('acquisition_domain','촬영 출처 도메인')]:
                for value,n in sorted(Counter(r[key] for r in rows).items(),key=lambda x:str(x[0])):
                    shown={'day':'주간','night':'야간','indoor':'실내','outdoor':'실외','mixed':'혼합'}.get(value,value) if value else MISSING
                    conditions.append(base|{'조건 항목':label,'메타데이터 값':shown,'영상 수':n})
    for kind in KINDS:
        train=[frame_records[fid] for fid in populations['TRAIN_'+kind]]
        train_ids={r['recording_id'] for r in train if r['recording_id'] is not None}
        train_hashes={r['image_sha256'] for r in train}
        for population in ('DEV292','PHYSICAL_FINAL_POSITIVE','PHYSICAL_FINAL_NEGATIVE'):
            target=[frame_records[fid] for fid in populations[population]]
            target_ids={r['recording_id'] for r in target if r['recording_id'] is not None}
            target_hashes={r['image_sha256'] for r in target}
            overlap=sorted(train_ids & target_ids)
            overlaps.append({'학습 종류':LABELS[kind],'비교 모집단':POP_LABELS[population],
                '명시 녹화 ID 교집합':','.join(overlap) if overlap else '없음',
                '녹화가 겹치는 학습 영상':sum(r['recording_id'] in overlap for r in train),
                '녹화가 겹치는 비교 영상':sum(r['recording_id'] in overlap for r in target),
                '학습 녹화 ID 없음':sum(r['recording_id'] is None for r in train),
                '비교 녹화 ID 없음':sum(r['recording_id'] is None for r in target),
                '선언된 이미지 SHA 교집합':len(train_hashes & target_hashes)})
    truth=[]
    for population in ('PHYSICAL_FINAL_POSITIVE','PHYSICAL_FINAL_NEGATIVE'):
        rows=[frame_records[fid] for fid in populations[population]]
        kinds=sorted({r['kind'] for r in rows}) or ['NONE']
        for kind in kinds:
            members=[r for r in rows if r['kind']==kind]
            claims=[e for r in members for e in r['independent_6d_truth_metadata_evidence']]
            truth.append({'모집단':POP_LABELS[population],'종류':LABELS[kind],'영상 수':len(members),
                '독립 6D 정답 근거':'메타데이터 명시 있음·본체 미검증' if claims else ('해당 없음' if not members else '없음·존재 여부 미확인'),
                '허용 근거 범위':'frames.csv, sessions.csv, session.json 및 contract만 확인; annotation·RGB 미열람'})
    meta.verify()
    protected_records=[dict(frame_id=r.get('frame_id'),
        image_sha256=r.get('image_sha256') or None,
        source_image_sha256=r.get('source_image_sha256') or None,
        source=str(WORK/'manifests/frames.csv')+'#frame_id='+r['frame_id']) for r in physical]
    cache=dict(schema='batch1_metadata_recordings_v1',created_utc=now(),frame_records=frame_records,populations=populations,
        protected_final_frame_ids=[r['frame_id'] for r in protected_records],
        protected_final_image_sha256=sorted({r[key] for r in protected_records
            for key in ('image_sha256','source_image_sha256') if r[key] is not None}),
        protected_final_records=protected_records,
        recording_registry_semantics='SOURCE_RECORDING_GROUPS.json의 명시 recording_id. 원 registry 방법은 이미지 내용 SHA 중첩과 union-find이며, 별도의 카메라 녹화 로그를 새로 인증하지 않았다.',
        recording_ci_eligibility={name:dict(eligible=all(frame_records[fid]['recording_id'] is not None for fid in ids),
            known=sum(frame_records[fid]['recording_id'] is not None for fid in ids),
            missing=sum(frame_records[fid]['recording_status']=='missing_metadata' for fid in ids),
            ambiguous=sum(frame_records[fid]['recording_status']=='ambiguous_conflicting_metadata' for fid in ids),
            rule='결측·상충이 있는 모집단의 녹화 단위 CI는 부분 확인 표본으로 대체 계산하지 않음') for name,ids in populations.items()},
        policy=dict(recording_id='명시된 메타데이터 ID 또는 registry exact session_key 대응만 사용; 세션명으로 추정하지 않음',
            indoor_outdoor='명시 조건만 사용; acquisition_domain이나 폴더명에서 추정하지 않음',
            missing='JSON null, 표의 없음',contents_read='메타데이터만; RGB·원본 annotation·모델·예측·점수 본체는 미열람'),
        sources=list(meta.files.values()))
    write(OUT/'cache/METADATA_RECORDINGS.json',cache)
    outputs=[]
    for name,rows in [('T0_POPULATIONS',overview),('T0_RECORDINGS',recordings),('T0_SESSIONS',session_counts),
                      ('T0_CONDITIONS',conditions),('T0_TRAIN_RECORDING_OVERLAP',overlaps),('T0_FINAL_TRUTH_METADATA',truth)]:
        dest=OUT/'tables'/(name+'.csv');csvwrite(dest,rows);outputs.append(dest)
        md=OUT/'tables'/(name+'.md')
        md.write_text('[확인] 허용된 메타데이터에서만 집계하였다. 없음은 추정하지 않은 결측이다.\n\n'+mdtable(list(rows[0]),[[r[k] for k in rows[0]] for r in rows]) if rows else '[확인] 해당 모집단에 등록된 행이 없다.\n')
        outputs.append(md)
    final_csv_audit={key:dict(rows=len(rows),frame_id_set_matches_authoritative_role=(
        {r['frame_id'] for r in rows}=={r['frame_id'] for r in physical if r['is_positive']==('true' if key=='POSITIVE' else 'false')}))
        for key,rows in final_manifest.items()}
    evidence=dict(schema='batch1_metadata_sources_v1',created_utc=now(),files=list(meta.files.values()),
        evidence=[dict(path=str(WORK/'README.md'),lines='56–89',meaning='FINAL_EVAL은 DEV 실행 alias이며 physical FINAL과 구분'),
            dict(path=str(WORK/'DATASET_CONTRACT.json'),json_pointer='/acquisition_domains',meaning='촬영 출처 범주는 실내·실외 환경 요인이 아니며 결측 추정 금지'),
            dict(path=str(REGISTRY),json_pointer='/groups',meaning='명시 녹화 ID와 session_key의 기존 alias 대응'),
            dict(path=str(REGISTRY),lines='4–18,2315–2345,3257–3265',meaning='이미지 SHA 중첩 기반 group 정의와 canonical eval_outside REC_012 / raw capturepallet02 REC_038 경로 문맥'),
            dict(path=str(WORK/'manifests/frames.csv'),lines='2816',meaning='한 프레임의 canonical 이미지 경로와 raw source_image_path를 동시에 명시한 상충 예시')],
        physical_final_definition='frames.csv의 population_role=FINAL을 양성/음성으로 분리. FINAL_EVAL alias나 ALL_AVAILABLE을 physical FINAL로 쓰지 않음.',
        physical_final_positive=len(populations['PHYSICAL_FINAL_POSITIVE']),physical_final_negative=len(populations['PHYSICAL_FINAL_NEGATIVE']),
        final_manifest_audit=final_csv_audit,missing_session_metadata_paths=sorted(session_metadata_missing),
        core_manifest_K_fields=sum(frame_records[fid]['manifest_has_K'] for fid in populations['DEV292']),
        independent_6d_truth='독립 외부 측정 정답의 존재를 허용된 메타데이터만으로 확정하지 못하면 없음·미확인으로 보고한다.',
        recording_ambiguities=[r for r in frame_records.values() if r['recording_status']=='ambiguous_conflicting_metadata'],
        original_annotation_files_opened=0,RGB_files_opened=0,model_files_opened=0,forwards=0,
        output_bindings=[dict(path=str(p),sha256=sha(p)) for p in outputs])
    write(OUT/'preflight/METADATA_SOURCES.json',evidence)
    summary='''# T0 메타데이터 집계\n\n[확인] DEV292, 종류별 학습 후보1,000장과 별도 PLASTIC184의 메타데이터를 사용하였다. 보호된 물리 FINAL은 frames.csv의 FINAL 역할만 집계하고 양성·음성을 분리하였다. FINAL_EVAL은 DEV 실행 alias이므로 물리 FINAL에 포함하지 않았다.\n\n[확인] 녹화 ID는 명시된 ID와 기존 registry의 exact session_key 대응만 사용했다. 없는 ID는 null/없음이며 세션 이름을 녹화 ID로 대체하지 않았다. 실내·실외도 명시 조건이 없으면 없음이다. acquisition_domain은 촬영 출처 범주로 별도 표기했다.\n\n[확인] 이미지 SHA 교집합은 CSV/manifest에 선언된 SHA의 비교이다. T0는 이미지 바이트를 새로 읽어 검증하지 않았다. RGB·annotation 본체·모델을 열지 않았고 추론이나 학습을 수행하지 않았다.\n\n[확인] 독립6D 정답은 역할명이나 annotation 완료 여부만으로 인정하지 않는다. 외부 측정/독립 자세 참조의 명시적 메타데이터 근거가 없으면 존재 여부 미확인으로 보고한다.\n\n'''
    for title,rows in [('모집단',overview),('녹화 ID별 장수',recordings),('촬영 조건',conditions),('학습 풀 녹화 겹침',overlaps),('물리 FINAL 정답 근거',truth)]:
        summary+='## '+title+'\n\n'+(mdtable(list(rows[0]),[[r[k] for k in rows[0]] for r in rows]) if rows else '등록된 행 없음\n')+'\n'
    summary+='\n[확인] 파일별 SHA와 메타데이터 근거는 preflight/METADATA_SOURCES.json, 개별 frame→녹화 매핑은 cache/METADATA_RECORDINGS.json에 기록했다.\n'
    summary+='\n[확인] 기존 registry는 이미지 내용 SHA 중첩과 union-find로 정의된 ID이다. 별도 카메라 녹화 로그를 새로 인증하지 않았다. eval_outside 일부 5장은 canonical 세션 경로가 REC_012, 원본 이미지 부모 경로가 REC_038(2장)·REC_040(1장)·REC_041(2장)로 연결되어 식별 불가(메타데이터 상충)로 보존했다. 어느 경로를 우선하지 않았으며 registry 자체의 오류로 단정하지 않는다. 결측·상충이 있는 모집단의 녹화 단위 CI는 확인된 일부 표본으로 대체 계산하지 않는다.\n'
    (OUT/'tables/T0_METADATA_REPORT.md').write_text(summary)
    log('step_complete',task='T0_메타데이터_집계',cache_sha256=sha(OUT/'cache/METADATA_RECORDINGS.json'),
        sources_sha256=sha(OUT/'preflight/METADATA_SOURCES.json'),core_frames=len(populations['DEV292']),
        physical_final_positive=len(populations['PHYSICAL_FINAL_POSITIVE']),physical_final_negative=len(populations['PHYSICAL_FINAL_NEGATIVE']))
    print('T0 완료: 메타데이터만 집계, RGB/annotation/모델 본체 열람0, forward0',flush=True)


if __name__=='__main__':
    main()
