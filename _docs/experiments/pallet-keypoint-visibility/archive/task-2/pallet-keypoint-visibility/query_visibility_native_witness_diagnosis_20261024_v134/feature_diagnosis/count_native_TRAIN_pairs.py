"""TRAIN data-defined physical-corner pair coverage; no fit/error-based mining."""
from pathlib import Path
import collections,hashlib,json
import numpy as np

H=Path(__file__).resolve().parent;P=H.parents[1]
PROTOCOL=H/'PROTOCOL_BEFORE_NATIVE_PAIR_COUNTS.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def stats(a):
    a=np.asarray(a,np.float64)
    return dict(N=len(a),mean=float(a.mean())if len(a)else None,quantile_0_25_50_75_100=np.quantile(a,[0,.25,.5,.75,1]).tolist()if len(a)else None)
def write(p,d):
    with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def run():
    pr=read(PROTOCOL);rr=rows(check(pr['source_inputs']['GT_ledger']))[::2]
    assert len(rr)==6144
    with np.load(check(pr['source_inputs']['features']),allow_pickle=False)as f:
        av=f['query_available'].copy();constructed=f['feature_constructed'].copy();q=f['query_xy'].copy()
    configs=read(check(pr['source_inputs']['base_configs']))['rows'];assert len(configs)==192
    target=np.array([0 if r['teacher_state']=='D'else 1 if r['teacher_state']=='E'else-1 for r in rr]);fold=np.array([r['split']for r in rr])
    pools=[np.flatnonzero((fold=='TRAIN')&av&(target==t))for t in(0,1)];assert list(map(len,pools))==[1869,131]
    groups=[]
    for base in range(128):
        for slot in range(8):
            ix=np.array([(base*4+v)*8+slot for v in range(4)]);ss=[rr[i]['teacher_state']for i in ix];aa=av[ix].tolist()
            assert all(rr[i]['split']=='TRAIN'and not rr[i]['correspondence_boundary']for i in ix)
            ce_nd=ss[1]=='E'and ss[2]=='D';eligible=ce_nd and aa[1]and aa[2]
            full=ss==['D','E','D','D']and all(aa)
            groups.append(dict(base_index=base,native_slot=slot,global_indices=ix.tolist(),states_BEFORE_COVER_NEAR_FAR=ss,
                available_BEFORE_COVER_NEAR_FAR=aa,constructed_BEFORE_COVER_NEAR_FAR=constructed[ix].tolist(),
                renderer_selected_native=slot==configs[base]['selected_CF_index'],factors=configs[base]['factors'],
                raw_coverE_nearD=ce_nd,available_coverE_nearD=bool(eligible),all4_D_E_D_D_and_available=bool(full),
                cover_near_q_movement_px=float(np.linalg.norm(q[ix[1]]-q[ix[2]]))if constructed[ix[1]]and constructed[ix[2]]else None))
    assert len(groups)==1024
    eligible=[g for g in groups if g['available_coverE_nearD']];full=[g for g in groups if g['all4_D_E_D_D_and_available']]
    hardD=np.array([g['global_indices'][2]for g in eligible]);pairE=np.array([g['global_indices'][1]for g in eligible])
    assert len(np.unique(hardD))==len(hardD)and len(np.unique(pairE))==len(pairE)
    trace=P/'query_visibility_native_geometry_control_20261024_v133/runs/C24_C25_MATCHED_G42_FIXED8000/C24_ZERO_GEOMETRY_CONTROL/TRAIN_DRAWS_AND_LOSSES.jsonl'
    draws=np.zeros(6144,np.int64);coocc=np.zeros(len(eligible),np.int64);batches_any=0
    for line in Path(trace).read_text().splitlines():
        r=json.loads(line);ii=np.asarray(r['native_occurrences'],np.int64);assert len(ii)==64
        draws+=np.bincount(ii,minlength=6144);unique=set(ii.tolist())
        present=np.array([int(e)in unique and int(d)in unique for e,d in zip(pairE,hardD)])
        coocc+=present;batches_any+=bool(present.any())
    assert draws.sum()==512000 and draws[pools[0]].sum()==draws[pools[1]].sum()==256000
    audit=[]
    paths=[('V129_C20_C21','query_visibility_registered_native_diversity_20261023_v129/source_fit_preparation/train_C20_C21_once.py'),
           ('V130_C22_C23','query_visibility_source_coverage_duration_20261024_v130/trainer_preparation/train_C22_C23_once.py'),
           ('V133_C24_C25','query_visibility_native_geometry_control_20261024_v133/source_fit_preparation/train_matched_geometry.py')]
    for phase,rel in paths:
        f=P/rel;txt=f.read_text();matches=[dict(line_number=i,text=line.strip())for i,line in enumerate(txt.splitlines(),1)
            if ('binary_cross_entropy_with_logits'in line and('ni'in line or phase=='V133_C24_C25'))or'nl=native_bce'in line or'def native_bce'in line or'no_native_paired_hinge'in line or'no_native_paired_hinge'in line]
        audit.append(dict(phase=phase,code=bind(f),explicit_native_pair_ranking=False,native_NEAR_SHAM_hardD_stratification=False,
            native_sampler='uniform independent32D32E overknownavailableTRAIN',native_loss='BCE only',
            C20_C22_native_loss='None in public-only control'if phase!='V133_C24_C25'else'BCE in both matchedarms',
            old_public_matched_hinge=phase=='V133_C24_C25',line_evidence=matches,
            note='V133 old192 hinge uses public/non-native cover/sham pairs; actualnative64 has no pairedhinge.'if phase=='V133_C24_C25'else'No nativepairrole is consulted by sampler/loss; source-emulated q corpus differs fromactualR0.'))
    def tally(gs):
        return dict(groups=len(gs),bases=len({g['base_index']for g in gs}),renderer_selected=sum(g['renderer_selected_native']for g in gs),
            by_material=dict(collections.Counter(g['factors']['material']for g in gs)),
            by_foreign_family=dict(collections.Counter(g['factors']['foreign_family']for g in gs)),
            by_geometry_family=dict(collections.Counter(g['factors']['geometry_family']for g in gs)),
            by_native_slot=dict(collections.Counter(str(g['native_slot'])for g in gs)))
    out=dict(completed=True,PASS=True,source_only=True,TRAIN_only_pair_definition=True,no_modelerror_or_VAL_mining=True,
        all_TRAIN_groups=1024,raw_coverE_nearD=tally([g for g in groups if g['raw_coverE_nearD']]),
        available_coverE_nearD=tally(eligible),available_full_D_E_D_D=tally(full),
        native_TRAIN_available_D=len(pools[0]),native_TRAIN_available_E=len(pools[1]),hard_nearD_unique_occurrences=len(hardD),paired_coverE_unique_occurrences=len(pairE),
        expected_uniform_draws_per_step_each_D=32/len(pools[0]),expected_uniform_draws_per_step_each_E=32/len(pools[1]),
        within_pair_E_vs_hardD_expected_exposure_ratio=len(pools[0])/len(pools[1]),
        actual8000_uniform_sampler=dict(all_D_draws=stats(draws[pools[0]]),all_E_draws=stats(draws[pools[1]]),hard_nearD_draws=stats(draws[hardD]),paired_coverE_draws=stats(draws[pairE]),
            hard_nearD_never_seen=int((draws[hardD]==0).sum()),same_pair_cooccurrence_batches_per_group=stats(coocc),batches_with_any_pair=batches_any,
            source_trace=bind(trace)),
        prior_loss_audit=audit,groups=groups,protocol=bind(PROTOCOL),code=bind(__file__),new_training_or_model_or_RGB_private_V112_PnP=0,
        interpretation_KO=['모든후보는 TRAIN의사전정의된같은physicalcorner D/E라벨·가용성으로만 구성한다. VAL오류나모델점수로pair를고르지않았다.',
            '기존 균등class32/32 sampling은개별E를개별D보다약14배자주노출하지만hardD도모두학습에노출됐다. 미관측이나노출비율만을실패원인으로확정할수없다.',
            '실제q가다른두입력의같은physicalcorner semanticD/E ranking감독은정의할수있다. 이것은동일q의순수가림인과대조와는다르다.',
            'Nativepairedranking·nearSHAMhardD층화는직전C20~C25에서실행하지않았다. 기존publicpairedhinge와혼동하지않는다. 새fit나loss선택은수행하지않았다.'])
    f=H/'NATIVE_TRAIN_PAIR_COVERAGE_AND_LOSS_HISTORY.json';write(f,out)
    print(json.dumps(dict(result=bind(f),available_pairs=tally(eligible),full_D_E_D_D=tally(full),hardD_draws=stats(draws[hardD]),pairE_draws=stats(draws[pairE])),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
