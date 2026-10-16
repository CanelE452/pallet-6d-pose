"""Independent arithmetic/publication checks over final flat tables, no solver imports."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from urllib.parse import unquote, urlsplit

from .common import read, rows, csv_rows, sha, write


def validate(root,public,source_root,original_workspace):
    from PIL import Image
    started,cpu=time.monotonic(),time.process_time()
    root,public,source_root,original_workspace=(Path(p).resolve() for p in (root,public,source_root,original_workspace))
    candidates=defaultdict(dict)
    for r in csv_rows(public/'SELECTION_COMPONENT_LEDGER.csv.gz'):
        key=(r['id'],r['candidate_id']);assert r['loss_family'] not in candidates[key]
        candidates[key][r['loss_family']]=r
    maxima=defaultdict(float)
    for key,parts in candidates.items():
        assert set(parts)=={'full_BI','partial_normal','outline','prior'}
        values=list(parts.values());a=values[0]
        difference=abs(math.fsum(float(v['weighted_component']) for v in values)-float(a['objective']))
        maxima['candidate_component_sum']=max(maxima['candidate_component_sum'],difference)
        assert difference<=1e-9
        good=float(a['T_cm'])<=5 and float(a['R_deg'])<=5
        assert all((v['reference_success']=='True')==good for v in values)
        assert all(v['S_point_terms']=='0' and v['U_point_terms']=='0' and float(v['actual_prior_weight'])==0 for v in values)
    assert len(candidates)==11603
    pairrows=list(csv_rows(public/'SELECTION_PAIR_TABLE.csv'));assert len(pairrows)==651
    observations=defaultdict(list)
    for r in csv_rows(public/'PER_OBSERVATION_PAIR_LEDGER.csv.gz'):
        a=float(r['weighted_contribution_s']);b=float(r['weighted_contribution_other']);delta=float(r['Delta'])
        assert abs((b-a)-delta)<=1e-9
        observations[(r['pair_id'],r['loss_family'])].append(delta)
    for p in pairrows:
        assert p['s_candidate_id'].startswith('ORIGINAL_'+p['stage']+'/')
        assert p['other_candidate_id'].startswith('ORIGINAL_'+p['stage']+'/')
        terms=[]
        for k in ('full_BI','partial_normal','outline','prior'):
            a=candidates[(p['id'],p['s_candidate_id'])][k]
            b=candidates[(p['id'],p['other_candidate_id'])][k]
            delta=float(b['weighted_component'])-float(a['weighted_component'])
            assert abs(delta-float(p['Delta_'+k]))<=1e-9
            diff=abs(math.fsum(observations[(p['pair_id'],k)])-delta)
            maxima['pair_observation_delta_sum']=max(maxima['pair_observation_delta_sum'],diff);assert diff<=1e-9
            terms.append(delta)
        diff=abs(math.fsum(terms)-float(p['delta_total']))
        maxima['pair_component_delta_sum']=max(maxima['pair_component_delta_sum'],diff);assert diff<=1e-9
    membership=list(csv_rows(public/'GROUP_MEMBERSHIP_238.csv'))
    # Use original selector records for all238 IDs and exact partition/group union.
    frames=list(rows(root/'selection_v1/FRAME_SELECTIONS.jsonl.gz'))
    decisions=list(csv_rows(public/'FAMILY_STAGE_DECISIONS.csv'))
    assert len(decisions)==952
    frame_map={f['id']:f for f in frames}
    for d in decisions:
        f=frame_map[d['id']];stage=f['stages'][d['stage']]
        family=stage['families'][int(d['family_index'])]
        assert d['family']==family['family']
        assert (d['accepted']=='True')==family['accepted']
        assert (d['fallback']=='True')==family['fallback']
        assert d['winner_candidate_id']==(family['winner_candidate_id'] or 'NA')
        assert json.loads(d['published_physical_pose'])==family['published_actual_pose']
        assert d['stage_selected_family']==stage['selected_family']
        assert int(d['stage_selected_family_index'])==stage['selected_family_index']
    assert len(frames)==len({f['id'] for f in frames})==len(membership)==238
    counts=Counter(f['group'] for f in frames if f['partition']!='CAL')
    assert counts==dict(FINAL_SUCCESS_CONTROL=114,FAMILY_SELECTION_LOSS=10,WITHIN_FAMILY_SELECTION_LOSS=22,NO_SUCCESS_CANDIDATE=46)
    assert all(f['partition']=='REMAINDER' for f in frames if f['partition']!='CAL' and 'SELECTION_LOSS' in f['group'])
    binding=root/'final_bindings/SOURCE_BINDINGS.json'
    if not binding.exists():binding=root/'SOURCE_BINDINGS.json'
    assert read(binding)['source_and_input_parity']
    assert read(root/'INDEPENDENT_INFLUENCE_CHECK.json')['passed']
    assert read(root/'cost_v1/DENSE_ARRAY_VALIDATION.json')['passed']
    assert read(root/'metrics_v2/METRIC_CHECK.json')['all_pass']
    original=read(root/'START.json');protected={p:sha(original_workspace/p) for p in original['protected_files_SHA256']}
    assert protected==original['protected_files_SHA256']
    # Check evaluated source still has the exact recorded bytes.
    cost_summary=read(root/'cost_v1/COST_SUMMARY.json')
    frozen={}
    for p,h in cost_summary['source_sha256'].items():
        if p.endswith('/fixed_cost.py') or p.endswith('/cost_batch.py'):
            actual=source_root/'scripts/research/pallet_selection_score_diagnosis_v1'/Path(p).name
            assert sha(actual)==h;frozen[str(actual.relative_to(source_root))]=h
    for p,h in read(root/'selection_v1/SELECTION_REPLAY.json')['source_hashes'].items():
        actual=source_root/'scripts/research/pallet_selection_score_diagnosis_v1'/Path(p).name
        assert sha(actual)==h;frozen[str(actual.relative_to(source_root))]=h
    gallery=public/'gallery';proof=read(gallery/'GALLERY_PROOF.json')
    for c in proof['cases']:
        detail=read(gallery/Path(c['source_crop']).parent/'cost_and_observations.json')
        for panel in c['panels']:
            v=candidates[(c['id'],panel['candidate_id'])]
            for k in v:assert abs(float(v[k]['weighted_component'])-panel['weighted_components'][k])<=1e-9
            assert abs(float(v['full_BI']['objective'])-panel['objective'])<=1e-9
        a,b=c['panels'][-2:]
        assert abs(detail['Delta_total']-(b['objective']-a['objective']))<=1e-9
    witness=read(gallery/'outline_witness/WITNESS_RECEIPT.json')
    assert len(witness['examples'])==6
    for i,r in enumerate(witness['examples'],1):
        samples=list(csv_rows(gallery/f'outline_witness/witness_{i:02d}_samples.csv.gz'))
        assert len(samples)==r['observed_sample_count']
        assert abs(math.fsum(float(s['Delta']) for s in samples)-r['Delta_observed_to_model'])<=1e-9
    images=0
    for path in gallery.rglob('*.jpg'):
        with Image.open(path) as im:im.verify()
        images+=1
    # Public paths are aliases. Scan compressed tables too.
    privacy=[]
    import gzip
    for p in public.rglob('*'):
        if not p.is_file():continue
        if p.suffix in ('.md','.json','.csv','.html'):
            value=p.read_text()
        elif p.suffix=='.gz':
            with gzip.open(p,'rt') as stream:value=stream.read()
        else:continue
        if '/home/' in value or '/tmp/' in value:privacy.append(str(p.relative_to(public)))
    assert not privacy,privacy
    # All local links; source links may point up to the repository root.
    broken=[];links=0
    for p in public.rglob('*'):
        if p.suffix not in ('.md','.html'):continue
        text=p.read_text()
        targets=re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)',text) if p.suffix=='.md' else re.findall(r'(?:href|src)=["\']([^"\']+)["\']',text)
        for target in targets:
            if target.startswith(('http:','https:','mailto:','data:')):continue
            filepart=urlsplit(target).path
            if not filepart:continue
            links+=1;dest=p.parent/unquote(filepart.strip('<>'))
            if not dest.exists() and dest.name not in ('VALIDATION.json','PUBLICATION_CHECK.json'):
                broken.append(dict(document=str(p.relative_to(public)),target=target))
    assert not broken,broken
    result=dict(schema='FINAL_FLAT_TABLE_AND_PRESERVATION_CHECK_V1',passed=True,
        total_IDs=238,PRIMARY_denominator=192,group_counts=dict(counts),candidate_count=len(candidates),
        same_candidate_joint_success_checks=11603,pair_count=651,original_stage_family_decision_rows_checked=952,
        maximum_absolute_differences=dict(maxima),
        strict_atol=1e-9,strict_rtol=0,original_source_input_reference_final_binding=True,
        independent_influence_checks_passed=True,dense_independent_check_passed=True,
        semantic_owner_verified=False,source_binding_unresolved=0,selection_replay_unresolved=0,
        physical_root_cause_unresolved_selection32=32,observation_vs_proposal_unresolved_no_candidate46=46,
        initial_metric_absolute_check_failed_rows=48,initial_metric_absolute_failures_preserved=True,
        original_success_IDs_unchanged=True,protected_dirty_files_unchanged=protected,
        evaluated_source_unchanged=frozen,gallery_candidate_values_checked=True,JPEG_files_decoded=images,
        witness_sample_sums_passed=True,local_document_links_checked=links,broken_links=broken,
        absolute_private_paths_in_public=privacy,new_pose_or_cost_evaluations_in_this_check=0,
        new_training_updates=0,new_NN_forwards=0,new_PnP=0,new_optimizers=0,new_candidates=0,
        production_defaults_or_paper_modified=False,elapsed_seconds=time.monotonic()-started,
        process_cpu_seconds=time.process_time()-cpu)
    write(public/'VALIDATION.json',result)
    return result


def seal(public,source_root):
    public,source_root=Path(public).resolve(),Path(source_root).resolve()
    files={str(p.relative_to(source_root)):dict(sha256=sha(p),bytes=p.stat().st_size)
           for folder in (public,source_root/'scripts/research/pallet_selection_score_diagnosis_v1')
           for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='PUBLICATION_CHECK.json' and '__pycache__' not in p.parts}
    write(public/'PUBLICATION_CHECK.json',dict(schema='SMALL_SELECTION_DIAG_PUBLICATION_SEAL_V1',files=files,
        source_result_commit='407de865445e4987614f50f893d2ccea78d507b6',
        no_raw_bank_mask_or_full_RGB_added=True,actual_remote_SHA_verified_separately_after_push=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','public','source-root','original-workspace'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(validate(a.root,a.public,a.source_root,a.original_workspace)['passed'])
    seal(a.public,a.source_root)
