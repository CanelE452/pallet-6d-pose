"""Fresh historical-val original groups, actual pixels and frozen-parent replay."""
from context31 import *
import prepare23 as P23
import features25
from cli_export import load_probe,load_population,read_label
from data30 import live_sources

def load_fresh(split,c,device='cpu'):
    folder=DATA/'cache'/split/c;m=read(folder/'manifest.json')
    pp=[torch.load(folder/p,weights_only=True,map_location='cpu') for p in m['chunks']]
    return m,{k:torch.cat([p[k] for p in pp]).to(device) for k in KEYS}

def register():
    RUN.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True)
    if (DATA/'PROTOCOL.json').exists():return
    meta=load_probe();old={x for rows in load_population().values() for x in rows}
    previous=[]
    for name in ('pallet_residual_v21','pallet_quality_v23'):
        previous+=read(ROOT/'data/pallet/results'/name/'SOURCES.json')['rows']
    excluded={r['pair_group_id'] for r in meta.values() if r['dataset_split']=='train'}|{meta[x]['pair_group_id'] for x in old}|{r['pair_group_id'] for r in previous}
    unique={}
    for r in sorted(meta.values(),key=lambda x:digest('v31-parent-unseen:'+x['pair_group_id'])):
        if r['dataset_split']=='val' and r['pair_group_id'] not in excluded:unique.setdefault(r['pair_group_id'],r)
    selected=list(unique.values())[:281];assert len(selected)==281
    prior=read(ROOT/'data/pallet/results/pallet_residual_v21/IMAGE_BINDINGS.json')
    forbidden=set(prior['forbidden'].values())|set(prior['selected'].values())|set(read(ROOT/'data/pallet/results/pallet_quality_v23/IMAGE_BINDINGS.json')['selected'].values())
    hashes={r['merged_stem']:sha(ROOT/r['image_path']) for r in selected}
    assert len(set(hashes.values()))==281 and not set(hashes.values())&forbidden
    pops={}
    for split,rows in [('new_train',selected[:153]),('new_check',selected[153:])]:
        pops[split]=[]
        for r in rows:
            ci=read_label(r['renderer_label_locator'])['camera_data']['intrinsics'];f=r['fixed_renderer_dimensions_m_xyz_model_input']
            pops[split].append(dict(id=r['merged_stem'],image=r['image_path'],padded=True,K=[[ci['fx'],0,ci['cx']],[0,ci['fy'],ci['cy']],[0,0,1]],dims=[f[0],f[2],f[1]],session=r['source'],pair_group_id=r['pair_group_id']))
    write(DATA/'POPULATIONS.json',pops);write(DATA/'SOURCES.json',dict(rows=selected))
    write(DATA/'PROTOCOL.json',dict(train=153,calibration=64,check=64,check_split='First64 new_check originals calibration, last64 heldout; grouped before extraction/training.',conditions=TRAIN_CONDITIONS,historical_YOLO_validation=True,not_whole_pipeline_untouched_holdout=True))
    write(DATA/'SPLIT_AUDIT.json',dict(status='PASS',unique_originals=281,unique_pair_groups=281,available_groups=len(unique),YOLO_train_group_overlap=0,prior_refinement_group_overlap=0,exact_image_overlap=0,selected_hashes=hashes,excluded_group_count=len(excluded),limits='Historical YOLO validation used in past model selection. Pair IDs and exact hashes audited; not certified unseen asset/scene or perceptual near-duplicate independence.'))
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,batch=16,updates_per_arm=2000,selector_updates=1000,correction_updates=1000,total_new_updates=36000,
        design='2x2 old153/new153 originals x legacy joint2000/separated selector1000+correction1000. New matched generic36direction control shares exact new selector. Old/new point-only selectors exposed as no-correction outputs. Selector trained once per dataset/seed; copied frozen into separated arms. 30000 arm-equivalent updates, actual optimizer updates27000 (selector reuse).',
        objective='Selector: bounded signed relative point error gain, no ranking loss; correction: independent signed amount, bounded mean incremental gain, ordered lower10% quantile, and separate rank logit. Rank picks only positive-lower-bound candidates. No condition labels in forward/loss.',
        budget_note='Same total per-path2000 update budget but selector/correction receive1000 each; package-level architecture/objective ablation, not isolated single-loss causality.',
        calibration='Fresh64 original groups diagnose lower-quantile coverage and ranking. No real-fit threshold or post-result checkpoint selection. Fresh final64 are evaluated after training fixed checkpoints.',
        acceptance='Every seed clean/crop10/20/30 within1% base; corner20/crop40 retain90%V18 gain. split_new vs generic_new and point-only selector crop20 >=1% eachseed and real9session95%upper<0. Real297 remains reusedDEV.',
        parents='V12/V25 and V29 DHT frozen; fresh groups disjoint YOLO train and V6/V21/V23 requested populations. Historical YOLO-val selection exposure remains.',
        not_executed='Physical visible edge/endpoint role GT supervision; new real final sessions; PCGrad.' ))
    # Correct explicit optimizer count: joint2*2000 + selector2*1000 + correction3*1000 =9000 per seed.
    p=read(RUN/'PROTOCOL.json');p['total_new_updates']=27000;write(RUN/'PROTOCOL.json',p)
    write(RUN/'PRIOR_BINDINGS.json',dict(files=read(OLD30.RUN/'COMPLETION.json')['files']))

def prepare():
    register();P23.RUN=DATA;P23.new_load=load_fresh
    P23.extract();P23.parents();P23.targets23()
    features25.RUN=RUN
    setup()
    for seed in SEEDS:
        for split in ('new_train','new_check'):
            for c in TRAIN_CONDITIONS:
                path=DATA/'sources'/split/c/f'seed{seed}.pt'
                if path.exists():continue
                m,O=load_fresh(split,c);O['edge']=features25.edges_for_manifest(m,split+'_'+c)
                r=torch.load(DATA/'parents'/split/c/f'seed{seed}.pt',weights_only=True,map_location='cpu')
                S=live_sources({k:v.cuda() for k,v in O.items()},{k:v.cuda() for k,v in r.items()},seed)
                path.parent.mkdir(parents=True,exist_ok=True);torch.save({k:v.cpu() for k,v in S.items()},path)
                print('SOURCES31',seed,split,c,len(O['box']),flush=True)
    write(RUN/'DATA_COMPLETE.json',dict(complete=True,GT_inference=False,split_audit_sha256=sha(DATA/'SPLIT_AUDIT.json'),target_audit_sha256=sha(DATA/'TARGET_AUDIT.json')))

if __name__=='__main__':prepare()
