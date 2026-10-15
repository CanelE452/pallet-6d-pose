"""Close actual historical population/count sources; no model or GT decoding.

The source245 CSV/archive hashes are checked as bytes, without parsing manual
corner JSON or decoding images. Namespaces and encoded-byte identity remain
explicit. Existing scientific records are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile


VIS=Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
ANNOT=Path('data/pallet/results/pallet_remaining_evidence_connection_20261006_v1/annotations')
OLD245=('test_reused_v1','region_completion_v1/TEST_N3','region_completion_v1/TEST_PF')
V32=VIS/'statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_pose_eval_v4_32_r2/PRIMARY_SCORE.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt')as f:return json.load(f)
    return json.loads(path.read_text())


def write(path,value):
    with Path(path).open('x')as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv_write(path,records,fields):
    with Path(path).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)


def resource_closeout(minimal_docs):
    folder=Path(minimal_docs);receipt=read(folder/'EXECUTION_RECEIPT.json')
    ledger=[json.loads(line)for line in (folder/'RESOURCE_LEDGER.jsonl').read_text().splitlines()]
    sums=defaultdict(float)
    for row in ledger:sums[row['kind']]+=row['elapsed_seconds']
    r=receipt['resources']
    for kind,key in [('gpu_process','instrumented_GPU_related_stage_elapsed_seconds'),
                     ('cpu_solver_analysis','instrumented_CPU_solver_analysis_stage_elapsed_seconds')]:
        if abs(sums[kind]-r[key])>1e-8:raise ValueError('Original resource ledger/receipt differs: '+kind)
    if (r['GPU_whole_process_containing_interval_upper_bound_seconds']>=r['GPU_cap_seconds'] or
        r['conservative_CPU_with_hybrid_GPU_jobs_elapsed_seconds']>=r['CPU_cap_seconds']):
        raise ValueError('Saved conservative resource bound exceeds the declared cap')
    return dict(status='CLOSED_WITH_DOCUMENTED_MEASUREMENT_EXCEPTION',
        source_sha256={n:sha(folder/n)for n in ('EXECUTION_RECEIPT.json','RESOURCE_LEDGER.jsonl','COST_AND_VALIDATION_KO.md')},
        original_ledger_entries=len(ledger),recomputed_kind_totals_seconds=dict(sums),
        duplicate_metadata_is_not_an_executed_stage=True,recorded_resource_fields=r,
        GPU_conservative_bound_within_cap=True,CPU_conservative_stage_account_within_cap=True,
        exact_initial_small_IO_startup_teardown_and_full_publication_wall_recoverable=False,
        exception='Fine-grained early reads/probes and process startup/teardown were not completely timed. The recorded captured interval is preserved; no missing seconds or exact end-to-end total are imputed.',
        full_detector_to_final_pose_time_measured=False,
        full_time_NA_is_allowed_by_instruction_line403=True,p_value_NA_is_allowed_by_line348=True,
        no_new_timing_or_model_experiment=True)


def run(*,source_root,current_inputs,archive_dir,minimal_docs,output,published_main=None):
    start=time.monotonic();source_root,archive_dir,output=map(Path,(source_root,archive_dir,output))
    if output.exists():raise FileExistsError('Preserve all previous closeout attempts')
    current=read(current_inputs)
    if current.get('reference_or_GT_inputs')is not False:raise ValueError('Predictor-only current input required')
    frames=current['frames'];ids=[r['id']for r in frames]
    if (len(ids)!=238 or len(set(ids))!=238 or
        Counter(r['partition']for r in frames)!=Counter(CAL=46,TEST=49,REMAINDER=143)):
        raise ValueError('Exact current fixed238 frame manifest required')
    primary=[r['id']for r in frames if r['partition']!='CAL'];test=[r['id']for r in frames if r['partition']=='TEST']
    image_hashes={r['image_sha256']for r in frames}
    sources={};pred=[];predcsv=[]
    for rel in OLD245:
        p=source_root/VIS/rel;seal=read(p/'PREDICTION_SEAL.json');file=p/'PREDICTIONS.jsonl.gz'
        if sha(file)!=seal.get('predictions_sha256',seal.get('sha256')):raise ValueError('245-row source seal changed')
        counts=Counter()
        with gzip.open(file,'rt')as stream:
            for line in stream:
                # These journals are prediction-only, with no reference fields.
                row=json.loads(line);counts[row['id'],row['method']]+=1
        unique=sorted({a for a,b in counts});methods=Counter()
        for (ident,method),n in counts.items():methods[method]+=n
        if (sum(counts.values())!=245 or len(unique)!=49 or len(methods)!=5 or
            any(v!=1 for v in counts.values()) or set(unique)!=set(test)):
            raise ValueError('Actual245 source must preserve49 current TEST IDs x5 methods')
        pred.append(dict(source=str(VIS/rel),rows=245,unique_frame_count=49,method_counts=dict(methods),
            ids=unique,old_not_TEST=sorted(set(unique)-set(test)),TEST_not_old=sorted(set(test)-set(unique)),
            old_not_PRIMARY=sorted(set(unique)-set(primary)),PRIMARY_not_old=sorted(set(primary)-set(unique)),
            prediction_sha256=sha(file),seal_sha256=sha(p/'PREDICTION_SEAL.json')))
        predcsv.extend(dict(source=str(VIS/rel),id=ident,method=method)for ident,method in sorted(counts))
        sources[str(VIS/rel/'PREDICTION_SEAL.json')]=sha(p/'PREDICTION_SEAL.json')
    v32=read(source_root/V32);arm='CLEAN_FRESH_BI_SOURCE_NATIVE'
    if [r['id']for r in v32['rows']]!=primary:raise ValueError('Old110 numerator source must have same192 IDs/order')
    success=[r['id']for r in v32['rows']if r['metrics'][arm]['translation_cm']<=5 and r['metrics'][arm]['rotation_deg']<=5]
    if len(success)!=110:raise ValueError('Fixed saved success numerator changed')
    sources[str(V32)]=sha(source_root/V32)
    manifest=source_root/ANNOT/'HANDOFF_SOURCE_CONNECTION.csv'
    with manifest.open()as f:original=list(csv.DictReader(f))
    records=[r for r in original if r['receiver_source_archive_member_status']=='EXACT_EXISTING_ARCHIVE_MEMBER']
    if (len(original)!=312 or len(records)!=245 or
        sum(r['receiver_git_blob_status']=='EXACT_EXISTING_GIT_BLOB'for r in original)!=67):
        raise ValueError('Existing312-source manifest partition changed')
    connection=read(source_root/ANNOT/'CONNECTION.json');archives=[];sourcecsv=[]
    for i,container in enumerate(connection['source_container_checks']):
        label='SOURCE_ARCHIVE_'+chr(65+i)
        name=container['absolute_path'].replace('\\','/').rsplit('/',1)[-1];path=archive_dir/name
        if sha(path)!=container['sha256']or path.stat().st_size!=container['bytes']:
            raise ValueError('Original source archive differs')
        rr=[r for r in records if r['original_source_path']==container['absolute_path']];imgs=[]
        with zipfile.ZipFile(path)as z:
            for r in rr:
                data=z.read(r['archive_member'])
                if hashlib.sha256(data).hexdigest()!=r['sha256']or len(data)!=int(r['bytes']):
                    raise ValueError('Original manual JSON byte hash differs; coordinates were not decoded')
                tokens=sorted(set(re.findall(r'20\d{6}[_T-]\d{6}',r['archive_member']+' '+r['receiver_identical_file_paths'])))
                sourcecsv.append(dict(source_id=r['source_id'],archive=label,
                    annotation_sha256=r['sha256'],annotation_bytes=int(r['bytes']),
                    archive_member_path_sha256=hashlib.sha256(r['archive_member'].encode()).hexdigest(),
                    recorded_session_tokens=';'.join(tokens),source_image_sha256=r['source_image_sha256'],
                    source_decoded_bgr_sha256=r['source_decoded_bgr_sha256']))
            for info in z.infolist():
                if not info.is_dir()and Path(info.filename).suffix.lower()in('.jpg','.png','.jpeg'):
                    imgs.append(hashlib.sha256(z.read(info)).hexdigest())
        if not {r['source_image_sha256']for r in rr}<=set(imgs):raise ValueError('Archived image byte hashes differ')
        archives.append(dict(archive=label,container_sha256=container['sha256'],container_bytes=container['bytes'],
            annotation_members_verified=len(rr),encoded_image_members_hashed=len(imgs),all_expected_image_hashes_found=True,
            recorded_session_tokens=sorted({s for r in sourcecsv if r['archive']==label for s in r['recorded_session_tokens'].split(';')if s})))
    hashes={r['source_image_sha256']for r in sourcecsv};overlap=sorted(hashes&image_hashes)
    if len(sourcecsv)!=245 or len(hashes)!=245 or overlap:raise ValueError('Distinct source245 identity facts changed')
    for n in ('HANDOFF_SOURCE_CONNECTION.csv','CONNECTION.json','inputs/annotations/verified/SOURCE_CONTAINERS.json'):
        sources[str(ANNOT/n)]=sha(source_root/ANNOT/n)
    result=dict(schema='ACTUAL_LEGACY_POPULATIONS_FINAL_CLOSEOUT_V1',status='CLOSED_ACTUAL_SOURCE_CATALOG_AND_ID_DIFFERENCES',
        fixed_current_frames=238,fixed_PRIMARY_frames=192,source_sha256=sources,
        current_predictor_input_sha256=sha(current_inputs),actual245_prediction_sources=pred,
        actual110_success_numerator=dict(source=str(V32),arm=arm,denominator=192,successes=110,
            same_current_PRIMARY_order=True,successful_ids=success,separate110_frame_evaluation_set=False),
        actual_distinct245=dict(original312_rows=312,git_source67=67,archive_source245=245,
            source_id_namespace='annotation_NNNNN, not current session:frame IDs',unique_image_hashes=245,
            archives=archives,current238_common_encoded_image_hashes=overlap,
            source_only_encoded_hashes=sorted(hashes-image_hashes),current_only_encoded_hashes=sorted(image_hashes-hashes),
            current_PRIMARY_common_encoded_hash_count=len(hashes&{r['image_sha256']for r in frames if r['partition']!='CAL'}),
            recorded_session_tokens_are_metadata_not_new_pixel_identity=True),
        identity_limit='Zero encoded-byte SHA overlap is not a proof of no re-encoded/cropped semantic frame overlap. Source containers/member hashes/session tokens authenticate the actual archived source; no new pixel decoding or scene matching is claimed.',
        scope_closure='The actual materials behind245-row,110-success and distinct245-source counts are catalogued and compared. No unsupported alternate245/110 evaluation population is created; the source/count requirement is closed on these real records.',
        published_main_observation=published_main,publication_field_is_caller_supplied_read_only_observation=True,
        new_manual_coordinate_JSON_decodes=0,new_image_decodes=0,new_model_mask_PnP_training_seed_or_GT_edits=0,
        scientific_source_or_old_seal_modified=False,wall_seconds=time.monotonic()-start)
    if published_main is not None and not re.fullmatch('[0-9a-f]{40}',published_main):raise ValueError('Full observed main SHA required')
    resource=resource_closeout(minimal_docs)
    output.mkdir(parents=True)
    write(output/'LEGACY_POPULATIONS_CLOSEOUT.json',result)
    write(output/'RESOURCE_MEASUREMENT_EXCEPTION.json',resource)
    csv_write(output/'DISTINCT245_SOURCES.csv',sourcecsv,list(sourcecsv[0]))
    csv_write(output/'PREDICTION245_ROWS.csv',predcsv,['source','id','method'])
    csv_write(output/'CURRENT238_IMAGES.csv',[{k:r[k]for k in ('id','partition','session','image_sha256')}for r in frames],
        ['id','partition','session','image_sha256'])
    write(output/'SEAL.json',dict(source_sha256=sha(__file__),assets={p.name:sha(p)for p in output.iterdir()if p.is_file()},
        no_scientific_prediction_or_reference_changed=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','current-inputs','archive-dir','minimal-docs','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--published-main')
    a=p.parse_args();r=run(source_root=a.source_root,current_inputs=a.current_inputs,archive_dir=a.archive_dir,
        minimal_docs=a.minimal_docs,output=a.output,published_main=a.published_main)
    print(json.dumps(dict(status=r['status'],current_frames=r['fixed_current_frames'],
        distinct_source245=r['actual_distinct245']['archive_source245'],new_model_or_PnP_calls=0),ensure_ascii=False))
