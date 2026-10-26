from common import *
def main():
    sources=[root/name for root in (FLIP,NOFLIP) for name in ('SUMMARY.json','SCORES.json','PREDICTIONS.json')]
    start('initial_preflight',sources)
    assert (OUT/'hashes/SNAPSHOT_BEFORE.json').is_file()
    evidence=[];identities=[]
    for root in (FLIP,NOFLIP):
        summary=read(root/'SUMMARY.json');scores=read(root/'SCORES.json');pred=read(root/'PREDICTIONS.json')
        assert set(scores['models'])==set(pred['predictions'])
        reference_ids=None;reference_points=None
        for model,ms in scores['models'].items():
            frames=ms['frames'];ids=[r['id'] for r in frames]
            assert len(ids)==292 and len(set(ids))==292
            assert ids==[r['id'] for r in pred['predictions'][model]]
            refs=[(r['id'],[(p['index'],p['xy'],p['source']) for p in r['all_annotated_reference']]) for r in frames]
            if reference_ids is None:reference_ids=ids;reference_points=refs
            assert ids==reference_ids and refs==reference_points
            for kind,nframe,npoint in [('PLASTIC',128,985),('WOOD',45,346),('GREEN',119,943)]:
                rows=[r for r in frames if kind_of(r)==kind]
                assert len(rows)==nframe and sum(len(r['all_annotated_reference']) for r in rows)==npoint
        rows=[r for r in scores['models']['R0_before_selftraining']['frames'] if kind_of(r)=='PLASTIC']
        points=[p for r in rows for p in r['all_annotated_reference']]
        correct=sum(p['error_px'] is not None and p['error_px']<=10 for p in points)
        obs=sum(p['error_px'] is not None for p in points)
        pointer='primary_diagonal' if root==FLIP else 'primary'
        assert correct==258 and len(points)==985 and obs==762
        assert summary[pointer]['PLASTIC']['models']['R0']['PCK10']==correct/len(points)
        evidence.append({'root':str(root),'models':len(scores['models']),'frames_per_model':292,'plastic_R0_correct10':correct,'plastic_R0_references':len(points),'plastic_R0_observed':obs,'summary_pointer':'/'+pointer+'/PLASTIC/models/R0/PCK10','input_sha256':{str(root/name):sha(root/name) for name in ('SUMMARY.json','SCORES.json','PREDICTIONS.json')}})
        identities.append(reference_points)
    assert identities[0]==identities[1]
    result={'initial_file_count_gate_pass':True,'summary_count_reproduction_exact':True,'cross_experiment_id_and_reference_identity_exact':True,'evidence':evidence,'T1b':'SKIPPED: 7x7/T=.1 is documented as training-loss extraction, not proven original inference contract; relocated evaluation source does not authenticate stated D0/P0 claim.','T3_actual_equivalence':'PENDING; no stage precision statistics authorized before all-frame verification.','T5_exception':'User explicitly allowed fixed maximum/median/minimum case selection only.','protected_FINAL_raw_opened':False,'forward_count':0,'new_optimizer_updates':0}
    write(OUT/'preflight/INITIAL_GATES.json',result)
    log('step_complete',task='initial_preflight',receipt_sha256=sha(OUT/'preflight/INITIAL_GATES.json'))
    print('INITIAL_FILE_AND_COUNT_GATES_PASS; PLASTIC_R0_258/985; T1b_SKIP; T3_EQUIVALENCE_PENDING')
if __name__=='__main__':main()
