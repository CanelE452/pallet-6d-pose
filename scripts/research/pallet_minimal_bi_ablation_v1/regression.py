"""Explicitly authorized pre-main gain10 success regression, never full scoring."""
from pathlib import Path
import json
import time
from . import scoring as S

OLD = S.B/'new_gain10_paired_BI_v2'
INPUT = S.B/'raw_ns_BI_gain10_input_v1/INPUTS.json'
OLD_SEAL = 'f57cc4e978f92b066d2179e1b5e5394c608685c3573a641fc297a29d788a7147'
ARM_PAIRS = {'R0':'RAW_YOLO_F','R1':'RAW_YOLO_BI_ONLY_FRESH_PNP',
             'N0':'N3_SUBPIX_F','N1':'N3_SUBPIX_BI_ONLY_FRESH_PNP'}


def skip_value(s, i):
    while s[i].isspace(): i += 1
    if s[i] in '[{':
        depth=0; quoted=False; escaped=False
        while i<len(s):
            c=s[i]
            if quoted:
                if escaped: escaped=False
                elif c=='\\': escaped=True
                elif c=='"': quoted=False
            elif c=='"': quoted=True
            elif c in '[{': depth+=1
            elif c in ']}':
                depth-=1
                if depth==0:return i+1
            i+=1
        raise ValueError('Malformed skipped object')
    if s[i]=='"':
        _,end=json.JSONDecoder().raw_decode(s,i)
        return end  # Only this scalar string, never skipped numeric pose values.
    while i<len(s) and s[i] not in ',}':i+=1
    return i


def object_slices(s, start=0):
    decoder=json.JSONDecoder();i=start
    while s[i].isspace():i+=1
    if s[i]!='{':raise ValueError('JSON object expected')
    i+=1;seen=set()
    while True:
        while s[i].isspace() or s[i]==',':i+=1
        if s[i]=='}':return
        key,i=decoder.raw_decode(s,i)
        if not isinstance(key,str) or key in seen:raise ValueError('Unique object keys required')
        seen.add(key)
        while s[i].isspace():i+=1
        if s[i]!=':':raise ValueError('Malformed JSON key')
        i+=1
        while s[i].isspace():i+=1
        end=skip_value(s,i)
        yield key,i,end
        i=end


def selected_truths(text, ids):
    """Keys are parsed, but numeric label bodies outside selected IDs are skipped."""
    result={};requested=set(ids)
    for key,start,end in object_slices(text):
        if key!='rows':continue
        for fid,a,b in object_slices(text,start):
            if fid in requested:
                result[fid]=json.loads(text[a:b])['target']['truth']
    if set(result)!=requested:raise ValueError('Every fixed gain10 reference required')
    return result


def run(*,source_root,parity_dir,manifest_path,output,root_regression_GO=False):
    if root_regression_GO is not True:raise PermissionError('Explicit regression-only GO before IO')
    wall=time.monotonic();cpu=time.process_time();source_root=Path(source_root);parity_dir=Path(parity_dir)
    if Path(output).exists():raise FileExistsError('Preserve every prior regression attempt')
    parity=S.read(parity_dir/'PARITY_CHECK.json')
    if parity['status']!='complete' or len(parity['records'])!=20 or parity['GT_inputs']is not False:
        raise ValueError('Completed twenty-observation and pose parity required before selected labels')
    old_dir=source_root/OLD;old_seal=S.read(old_dir/'PREDICTION_SEAL.json')
    if S.sha(old_dir/'PREDICTION_SEAL.json')!=OLD_SEAL or S.sha(old_dir/'PREDICTIONS.jsonl.gz')!=old_seal['predictions_sha256']:
        raise ValueError('Original fixed gain10 prediction seal changed')
    inputs=S.read(source_root/INPUT);ids=inputs['ordered_ids']
    if len(ids)!=10 or len(set(ids))!=10 or old_seal['ordered_ids']!=ids:
        raise ValueError('Same fixed posthoc gain10 required')
    manifest=S.read(manifest_path);binding=manifest['reference_bindings']['TEST']
    reference=source_root/binding['path']
    if S.sha(reference)!=binding['sha256']:raise ValueError('Fixed TEST reference bytes changed')
    truth=selected_truths(reference.read_text(),ids)  # Only ten numeric label records decoded.
    records=[];failures=[]
    for i,frame in enumerate(inputs['frames']):
        fid=frame['id'];old=S.read(old_dir/'frames'/f'{i:02d}_PREDICTION.json')
        if old['id']!=fid:raise ValueError('Original source order changed')
        new={}
        for branch,arm in (('RAW_YOLO','R1'),('N3_SUBPIX','N1')):
            saved=S.read(parity_dir/f'{i:02d}_{branch}.json')
            if saved['id']!=fid or saved['branch']!=branch:raise ValueError('Parity source order changed')
            new[arm]=saved['pose_result']['actual_pose']
        new['R0']=frame['arms']['RAW_YOLO']['coarse_pose'];new['N0']=frame['arms']['N3_SUBPIX']['coarse_pose']
        metrics={}
        for arm in S.ARMS:
            a=S.metric_for_pose(new[arm],truth[fid])
            b=S.metric_for_pose(old['predictions'][ARM_PAIRS[arm]]['actual_pose'],truth[fid])
            same=all(a[k]==b[k] for k in ('success_le5','success_strict_lt5'))
            if not same:failures.append(dict(id=fid,arm=arm,reason='Success classification differs'))
            metrics[arm]=dict(new=a,original=b,success_classification_exact=same)
        records.append(dict(id=fid,metrics=metrics))
    counts={a:sum(r['metrics'][a]['new']['success_strict_lt5']for r in records)for a in S.ARMS}
    expected=dict(R0=0,R1=7,N0=10,N1=10)
    if counts!=expected:failures.append(dict(reason='Historical gain10 regression counts differ',actual=counts,expected=expected))
    result=dict(schema='MINIMAL_BI_FIXED_GAIN10_PREMAIN_SUCCESS_REGRESSION_V1',
        status='PASS'if not failures else'FAIL',failure_count=len(failures),failures=failures,
        rows=10,arm_rows=40,subset_posthoc_not_population_generalization=True,
        original_and_new_success_classifications_exact=not failures,strict_counts=counts,
        inclusive_counts={a:sum(r['metrics'][a]['new']['success_le5']for r in records)for a in S.ARMS},
        records=records,selected_numeric_reference_rows_decoded=10,unselected_reference_pose_values_decoded=0,
        main_population_scored=False,thresholds_changed=False,
        source_sha256={Path(__file__).name:S.sha(__file__),'scoring.py':S.sha(S.__file__)},
        reference_binding=binding,manifest_sha256=S.sha(manifest_path),
        parity_sha256=S.sha(parity_dir/'PARITY_CHECK.json'),old_prediction_seal_sha256=OLD_SEAL,
        elapsed_wall_seconds=time.monotonic()-wall,process_CPU_seconds=time.process_time()-cpu)
    S.write(output,result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--parity-dir',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--root-regression-GO',action='store_true')
    a=p.parse_args();r=run(source_root=a.source_root,parity_dir=a.parity_dir,manifest_path=a.manifest,
        output=a.output,root_regression_GO=a.root_regression_GO)
    print(json.dumps({k:r[k]for k in('status','failure_count','strict_counts','inclusive_counts','elapsed_wall_seconds','process_CPU_seconds')}))
    if r['failure_count']:raise SystemExit(1)
