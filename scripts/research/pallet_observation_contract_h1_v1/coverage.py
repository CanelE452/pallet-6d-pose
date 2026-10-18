"""Saved-synthetic provenance, internal grouping and coverage diagnostics.

Provenance extraction never opens SYNTH_GT_LABELS or selector success/error
labels. Only explicit source identity fields survive the record whitelist.
"""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,math,re,resource,time
from pathlib import Path
from scripts.research.pallet_risk_information_v1.common import read,rows,sha,write,save_rows

PROVENANCE_KEYS=('idx','usable_id','seed','attempt_frame_index','attempt_seed','proposal_index','usable_slot',
    '_src_shard','_src_id','_src_root','scene_preset','background_asset','completion_mode')
SPLIT_SALT='OBSERVATION_CONTRACT_H1_POLICY_V1'

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def extract_provenance(repo,split_path,output):
    start,cpu=time.monotonic(),time.process_time();repo,split_path,output=map(Path,(repo,split_path,output));output.mkdir(parents=True,exist_ok=False)
    split=read(split_path);selected={int(r['id'].rsplit('f',1)[1]):r for r in split['all_rows']}
    if len(selected)!=1003:raise ValueError('Exact original1003 IDs required')
    mp=repo/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json';manifest=read(mp)
    if sha(mp)!=split['original_source_sha256']:raise ValueError('Original source manifest differs')
    source={r['id']:r for r in manifest['records']};rp=repo/'data/pallet/training_data/paper_release/v2_prod40k_clean_merged/records.jsonl'
    found={};digest=hashlib.sha256();record_count=0;index_counts=collections.Counter()
    # Skip unmatched JSON values entirely: decode only the1003 selected source
    # records, then retain generation identity fields, not visibility or labels.
    with rp.open('rb')as f:
        for line in f:
            digest.update(line);record_count+=1
            # idx is the original per-shard attempt index and can repeat.
            # usable_id is the merged dataset filename index; preserve both.
            matches=re.findall(rb'"usable_id"\s*:\s*(\d+)',line)
            if len(matches)!=1:raise ValueError('Missing/ambiguous original merged usable_id')
            index=int(matches[0]);index_counts[index]+=1
            if index not in selected:continue
            if index in found:raise ValueError('Duplicate original merged frame index')
            raw=json.loads(line);s=selected[index];m=source[s['id']]
            if raw['usable_id']!=index:raise ValueError('Original merged usable_id differs')
            keep={k:raw[k]for k in PROVENANCE_KEYS if k in raw}
            found[index]=dict(id=s['id'],split=s['split'],source_asset=s['source_asset'],scenario_id=s['scenario_id'],
                source=m['source'],source_kind=m['source_kind'],source_split=m['source_split'],
                source_image_sha256=m['image_sha256'],prepared_label_sha256=m['label_sha256'],
                renderer_annotation_locator_provenance_only=m['renderer_annotation_locator_provenance_only'],
                original_record_line_sha256=hashlib.sha256(line).hexdigest(),generation=keep,
                generation_attempt_key=canonical({k:keep.get(k)for k in('_src_root','_src_shard','attempt_seed')}),
                independence_status='GENERATION_IDENTITY_AVAILABLE_SEMANTICS_REQUIRES_SOURCE_REVIEW')
    if set(found)!=set(selected)or any(v!=1 for v in index_counts.values()):raise ValueError('Original record index coverage/uniqueness differs')
    result=[found[int(r['id'].rsplit('f',1)[1])]for r in split['all_rows']];path=output/'PROVENANCE_ROWS.jsonl.gz';save_rows(path,result)
    summaries={}
    for sp in('train','validation'):
        a=[r for r in result if r['split']==sp]
        summaries[sp]=dict(frames=len(a),assets=dict(collections.Counter(r['source_asset']for r in a)),
            generation_attempt_groups=len({r['generation_attempt_key']for r in a}),
            source_shards=dict(collections.Counter(r['generation'].get('_src_shard')for r in a)),
            missing_attempt_seed=sum(r['generation'].get('attempt_seed')is None for r in a),
            duplicate_image_groups=sum(v>1 for v in collections.Counter(r['source_image_sha256']for r in a).values()))
    receipt=dict(schema='H1_GUARD_LABEL_FREE_PROVENANCE_V1',status='EXTRACTED_GROUP_SEMANTICS_NOT_YET_APPROVED',
        rows=1003,original_records=record_count,provenance_rows_sha256=sha(path),split_sha256=sha(split_path),
        original_source_manifest_sha256=sha(mp),original_records_sha256=digest.hexdigest(),summary=summaries,
        retained_generation_keys=list(PROVENANCE_KEYS),supervision_label_files_opened=0,real_reference_files_opened=0,
        note='Same asset does not imply the same generation attempt. Unique seed identity alone does not prove independent scene-reset semantics; grouping review must be sealed before calibration labels.',
        source_sha256=sha(__file__),cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
    write(output/'PROVENANCE_SEAL.json',receipt);return receipt

def split_groups(provenance,*,group_mode,group_semantics_evidence):
    """Deterministic group-hash split, with outcome-free provenance only."""
    if group_mode not in('generation_run','generation_attempt','shared_asset_fallback'):raise ValueError('Explicit group contract required')
    if group_mode=='generation_attempt'and not group_semantics_evidence.get('generation_attempt_semantics_supported'):
        raise ValueError('Generation grouping requires actual source/provenance evidence')
    validation=[r for r in provenance if r['split']=='validation']
    if len(validation)!=235:raise ValueError('Original validation235 membership required')
    groups=collections.defaultdict(list);uncertain=[]
    def group_key(r):
        if group_mode=='generation_attempt':return r['generation_attempt_key']
        if group_mode=='shared_asset_fallback':return canonical([r['source'],r['source_asset']])
        g=r['generation'];root=g.get('_src_root');run=g.get('_src_shard')or root
        if not root or not run:uncertain.append(r['id'])
        return json.dumps(dict(source_root=root,generation_run=run),sort_keys=True,separators=(',',':'))
    for r in validation:
        groups[group_key(r)].append(r)
    ordered=sorted(groups,key=lambda g:hashlib.sha256((SPLIT_SALT+'|'+g).encode()).hexdigest())
    total=len(validation);prefix=0;candidates=[]
    for cutoff in range(1,len(ordered)):
        prefix+=len(groups[ordered[cutoff-1]])
        candidates.append((abs(3*prefix-2*total),cutoff))
    cutoff=min(candidates)[1]if candidates else len(ordered)
    dev=set(ordered[:cutoff])
    confirmation=set(groups)-dev;available=bool(dev and confirmation and not uncertain);assign=[]
    for r in validation:
        key=group_key(r)
        assign.append(dict(id=r['id'],group=key,subset='development'if key in dev else'confirmation'))
    return dict(schema='H1_GUARD_LABEL_FREE_INTERNAL_SPLIT_V1',status='SEALED_INTERNAL_GROUP_SPLIT'if available else'INTERNAL_POLICY_CHECK_UNAVAILABLE',
        group_mode=group_mode,group_semantics_evidence=group_semantics_evidence,hash_salt=SPLIT_SALT,
        rule='Sort canonical run IDs by SHA256(salt|canonical_run_ID). Choose group prefix minimizing abs(3*development_frame_count-2*235), ties smallest cutoff. Only frame counts/provenance, no outcome labels. Empty side or unresolved provenance disables adoption.',
        hash_ordered_groups=ordered,selected_prefix_cutoff=cutoff,
        frames=235,effective_groups=len(groups),independent_asset_groups=len({r['source_asset']for r in validation}),
        development_frames=sum(r['subset']=='development'for r in assign),confirmation_frames=sum(r['subset']=='confirmation'for r in assign),
        development_groups=len(dev),confirmation_groups=len(confirmation),assignments=assign,uncertain_provenance_ids=sorted(set(uncertain)),labels_read_for_split=0,
        limitation='Upstream validation235 already selected original model checkpoints. Internal split is not an independent validation or safety certificate.')

def seal_policy_split(provenance_dir,plan_path,output):
    provenance_dir,plan_path,output=map(Path,(provenance_dir,plan_path,output));plan=read(plan_path)
    if plan['status']!='FROZEN_BEFORE_FIRST_OPTIMIZER_UPDATE':raise ValueError('Frozen parent experiment plan required')
    pp=provenance_dir/'PROVENANCE_ROWS.jsonl.gz';ps=provenance_dir/'PROVENANCE_SEAL.json';seal=read(ps)
    if sha(pp)!=seal['provenance_rows_sha256']:raise ValueError('Label-free provenance bytes changed')
    evidence=dict(generation_run_identity_supported=True,
        source='Actual original records _src_shard, with explicit replacement _src_root fallback; global usable_id joins original merged filename.',
        approval='Root approved conservative generation-RUN grouping before calibration labels; same-asset is not automatic same-scene grouping.',
        independence='Run grouping keeps all potentially adjacent frames in one source run on one side. Independent scene-reset semantics and independent assets are not claimed.',
        original_records_sha256=seal['original_records_sha256'],provenance_seal_sha256=sha(ps))
    result=split_groups(list(rows(pp)),group_mode='generation_run',group_semantics_evidence=evidence)
    result.update(experiment_plan_sha256=sha(plan_path),provenance_rows_sha256=sha(pp),source_sha256=sha(__file__))
    write(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('provenance','split'));p.add_argument('--repo-root',type=Path);p.add_argument('--split-lock',type=Path);p.add_argument('--provenance-dir',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.command=='provenance':
        r=extract_provenance(a.repo_root,a.split_lock,a.output);print(json.dumps({k:r[k]for k in('status','rows','summary','cost')}))
    else:
        r=seal_policy_split(a.provenance_dir,a.plan,a.output);print(json.dumps({k:r[k]for k in('status','effective_groups','development_frames','confirmation_frames','development_groups','confirmation_groups')}))
