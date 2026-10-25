"""Prospective strong point / expected-part weak contract. No file access."""
import math
from collections import Counter
import numpy as np

META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
DOMAINS=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816')
STRONG_AUTHORITY='SYNTH_RENDERED_OWN_IMAGE_POINT0_TARGET_ALONE_D_COMBINED_DE'
WEAK_AUTHORITY='ASSISTANT_CONSENSUS_EXPECTED_FINITE_PART_SOURCE_GT_QUERY'

def need(ok,message):
    if not bool(ok):raise ValueError(message)

def context32(q,L,k,available,q8,L8,av8):
    """Same C61 information: native8 + self-masked other8 + F64/clip/FP32 XY16."""
    need(type(k) is int and 0<=k<8 and len(q8)==len(L8)==len(av8)==8,'native/group8')
    need(type(available) is bool and all(type(a) is bool for a in av8) and available==av8[k],'availability provenance')
    out=np.zeros(32,np.float32);out[k]=1
    if available:
        aq=np.asarray(q,np.float64);need(aq.shape==(2,) and np.isfinite(aq).all() and math.isfinite(L) and L>0,'finite active q/L')
        for j in range(8):
            if j==k or not av8[j]:continue
            z=np.asarray(q8[j],np.float64);need(z.shape==(2,) and np.isfinite(z).all() and L8[j] is not None and math.isfinite(L8[j]) and L8[j]>0,'finite available other')
            out[8+j]=1;out[16+2*j:18+2*j]=np.clip((z-aq)/np.float64(L),-8,8).astype(np.float32)
    return out

def weak_plan(rows):
    need(len(rows)==360 and len({r['row_id'] for r in rows})==360,'all360 immutable family rows')
    bygroup={};dev=set();result=[]
    for r in rows:
        g=r['generated_input'];need(r['source_role']==g['source_role'] and r['row_id']==g['row_id'],'weak identity/role')
        need(g['family']=='PUBLIC_SOURCE_GT_QUERY_JITTER_V1' and g['q8_origin']=='GT_DERIVED_TRAINING_INITIALIZATION','explicit GT-derived training query origin')
        need(g['training_admission'] is False and r['training_admission'] is False,'new root admission remains separate')
        bygroup.setdefault(g['group_id'],[]).append(r)
        if r['source_role']=='SOURCE_DEV':dev.add(g['component_id'])
    for rr in bygroup.values():need(sorted(r['generated_input']['native_slot'] for r in rr)==list(range(8)),'complete group8')
    for r in rows:
        g=r['generated_input'];gg=sorted(bygroup[g['group_id']],key=lambda x:x['generated_input']['native_slot']);other=[x['generated_input'] for x in gg]
        cc=context32(g['generated_query_xy'],g['L'],g['native_slot'],g['generated_query_available'],[x['generated_query_xy'] for x in other],[x['L'] for x in other],[x['generated_query_available'] for x in other])
        need(np.array_equal(cc,np.asarray(r['deployable_condition32'],np.float32)),'same C61 condition32, no ROI/pose/reference extra feature')
        known=r['weak_loss_candidate'] is True
        t=-1
        if known:
            need(r['target_state'] in ('D_VISIBLE_EXPECTED_PART_WEAK','E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK'),'finite expected-part D/E only')
            need(g['generated_query_available'] is True and all(r['support_checks'][k] is True for k in ('query_available','nominal_target_in_frame','nominal_target_in_crop96','nominal_target_strict_valid')),'support checked weak target')
            t=int(r['target_state']=='E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK');need(r['weak_target_E_binary']==t,'weak scalar meaning')
        fit=known and r['source_role']=='SOURCE_FIT'
        need(fit==(r['weak_loss_fit_candidate'] is True),'no DEV loss candidate')
        result.append(dict(row_id=r['row_id'],component_id=g['component_id'],source_role=r['source_role'],native_slot=g['native_slot'],query_xy=g['generated_query_xy'],L=g['L'],original_query_available=g['generated_query_available'],
            condition32=cc.tolist(),target=t,fit_eligible=fit,authority=WEAK_AUTHORITY,RGB96=g['RGB96'],valid96_RLE=g['valid96_RLE'],GT_derived_q8=True,training_admission=False))
    need(len(dev)==4,'four fixed DEV components')
    need(Counter(x['target'] for x in result if x['fit_eligible'])=={0:39,1:6},'frozen weak FIT45 candidate count')
    need(Counter(x['target'] for x in result if x['source_role']=='SOURCE_DEV' and x['target']>=0)=={0:9,1:3},'frozen weak DEV12 candidate count')
    return result,dev

def fit_mask(metadata,dev_components):
    """Original role never rewritten; missing lineage excluded, not assumed TRAIN."""
    return np.asarray([r.get('component_role')=='TRAIN' and r.get('nominal_fit_eligible') is True and
        isinstance(r.get('component_id'),str) and len(r['component_id'])==64 and r['component_id'] not in dev_components and
        r.get('lineage_bound') is True for r in metadata],bool)

def old_pools(caches,masks):
    """Original five tasks, new masked class normalization; no frozen253/314 assertion."""
    s,p,c=(caches[d] for d in DOMAINS);av=s['x']['original_query_available']&masks[DOMAINS[0]]; t=s['y']['support_target']
    support=[[np.flatnonzero(av&(t[:,j]==label)) for label in [1,0]] for j in range(2)]
    mass=s['y']['owner_soft'];ok=av&np.isfinite(mass).all(1);w=np.where(ok[:,None],np.nan_to_num(mass,nan=0),0.)
    need((w.sum(0)>0).all(),'all masked synthetic owner classes remain')
    conditional=[w[:,j]/w[:,j].sum(dtype=np.float64) for j in range(3)];hard=[]
    for name,cache,classes in [(DOMAINS[1],p,range(3)),(DOMAINS[2],c,range(2))]:
        need((cache['y']['support_target']==-1).all(),'public/CF owner never native/support labels')
        hard.append([np.flatnonzero(masks[name]&cache['x']['original_query_available']&(cache['y']['owner_target']==j)) for j in classes])
    need(all(len(x)>0 for group in support+hard for x in group),'every original task class remains; no fallback')
    return dict(support=support,conditional=conditional,public=hard[0],CF=hard[1])

def balanced_pools(target,mask):
    out=[np.flatnonzero(mask&(target==k)) for k in (0,1)];need(all(len(x)>0 for x in out),'both D/E classes, no pseudo-label fallback');return out

def draw_balanced(rng,pools,n):
    return np.concatenate([rng.choice(p,n,replace=True) for p in pools])

def balanced_loss(logits,target):
    import torch.nn.functional as F
    need(logits.shape==target.shape and all(bool((target==k).any()) for k in (0,1)),'known balanced loss')
    return sum(F.binary_cross_entropy_with_logits(logits[target==k],target[target==k].to(logits.dtype)) for k in (0,1))*.5

def objective(out,point,weak,weak_weight,five_losses):
    base,parts=five_losses({k:out[k][:192] for k in ('support_logits','owner_logits')})
    strong=balanced_loss(out['direct_end_external_logit'][192:224],point)
    local=balanced_loss(out['direct_end_external_logit'][224:],weak) if weak is not None and weak_weight>0 else strong.new_zeros(())
    return base+strong+weak_weight*local,parts,strong,local


CF_AUTHORITY='CONTROLLED_EXPECTED_FINITE_PART_WEAK_NOT_PHYSICAL_NATIVE_POINT_GOLD'
ARMS=('C68SHARED_POINT_PART_LOGIT','C69SEPARATE_POINT_PART_LOGITS')
VARIANTS=('BEFORE','COVER','SHAM')


def same_binding(a,b):
    return isinstance(a,dict) and isinstance(b,dict) and a.get('path')==b.get('path') and a.get('sha256')==b.get('sha256')


def same_numeric(a,b):
    return np.array_equal(np.asarray(a,np.float64),np.asarray(b,np.float64),equal_nan=True)


def cf_plan(rows,weak,original360):
    """Preserve144 rows; admit only complete constructed source-weak triples after root admission.

    BEFORE-only factual rows from exhausted attempts remain metadata, never a fabricated pair.
    No old original-query9/native target, ROI or GT geometry becomes a model feature.
    """
    need(len(rows)==144 and len({r['row_id'] for r in rows})==144,'all48x3 prospective rows')
    parents={r['row_id']:r for r in weak if r['target']==0}
    need(len(parents)==48,'exact prior GT-family D48, not original-query9')
    original={r['row_id']:r['generated_input'] for r in original360}
    bygroup={}
    for g in original.values():bygroup.setdefault(g['group_id'],[]).append(g)
    groups={}
    for r in rows:
        p=r['parent_row_id'];need(p in parents,'exact canonical D48 parent')
        v=r['variant'];need(v in VARIANTS and r['row_id']==p+'__'+v,'declared variant identity')
        need(v not in groups.setdefault(p,{}),'unique parent variant');groups[p][v]=r
        w=parents[p];g=original[p]
        need(same_binding(r['parent_RGB96_binding'],w['RGB96']),'exact prior parent provenance')
        need(r['source_role']==w['source_role'] and r['component_id']==w['component_id'],'immutable source role/component')
        need(type(r['native_slot']) is int and r['native_slot']==w['native_slot'],'original native ID')
        need(same_numeric(r['query_xy'],w['query_xy']) and same_numeric(r['L'],w['L']) and r['query_available'] is w['original_query_available'],'same query/L/U')
        need(r['valid96_RLE']==w['valid96_RLE'] and np.array_equal(np.asarray(r['condition32'],np.float32),np.asarray(w['condition32'],np.float32)),'same valid and exact original condition32')
        gg=sorted(bygroup[g['group_id']],key=lambda x:x['native_slot'])
        need([x['native_slot'] for x in gg]==list(range(8)) and r['q8_native_slots']==list(range(8)),'original complete native8 order')
        need(same_numeric(r['q8_xy'],[x['generated_query_xy'] for x in gg]) and r['q8_available']==[x['generated_query_available'] for x in gg],'original GT-derived training q8, not new geometry')
        need(r['authority']==CF_AUTHORITY and r['native_point_DE_target']==-1 and r['training_admission'] is False,'weak finite part separate from point gold/root admission')
        need(type(r['constructed']) is bool,'explicit construction status')
        need(type(r['input_available']) is bool and type(r['target_known']) is bool and r['input_available']==isinstance(r['RGB96'],dict),'input availability separate from label authority')
        need(r['target_known']==(r['weak_finite_part_target_D0_E1'] in (0,1)),'missing variants have unknown targets')
        need(same_numeric(r['q8_L'],[x['L'] for x in gg]),'same original training group8 scales')
    need(set(groups)==set(parents),'all48 parents retained, failures never dropped')
    pairs=[];statuses=[]
    for p in sorted(groups):
        rr=groups[p];need(set(rr)==set(VARIANTS),'all3 status rows')
        need(len({r['paired_id'] for r in rr.values()})==1,'same paired identity')
        need(rr['COVER']['constructed'] is rr['SHAM']['constructed'],'no half-constructed pair')
        active=rr['COVER']['constructed']
        before=rr['BEFORE'];need(isinstance(before['RGB96'],dict) and before['RGB96']['sha256']==parents[p]['RGB96']['sha256'] and before['target_known'] and before['weak_finite_part_target_D0_E1']==0,'BEFORE byte-exact role-specific copy of prior D parent even if geometry fails')
        if active:
            for v,t in zip(VARIANTS,(0,1,0)):
                r=rr[v];need(r['constructed'] and isinstance(r['RGB96'],dict) and r['weak_finite_part_target_D0_E1']==t,'complete weak D/E/D only')
            need(before['query_available'] is True,'no unavailable weak supervision')
            pairs.append({'parent_row_id':p,'paired_id':before['paired_id'],'component_id':before['component_id'],'source_role':before['source_role'],'fit_eligible':before['source_role']=='SOURCE_FIT','rows':[rr[v] for v in VARIANTS]})
        else:
            for v in ('COVER','SHAM'):
                need(rr[v]['RGB96'] is None and rr[v]['weak_finite_part_target_D0_E1'] is None,'exhausted variants not invented/labeled')
        statuses.append({'parent_row_id':p,'source_role':before['source_role'],'constructed_pair':active,'fit_eligible':active and before['source_role']=='SOURCE_FIT'})
    need(all(x['source_role'] in ('SOURCE_FIT','SOURCE_DEV') for x in statuses),'declared FIT/DEV only')
    return pairs,statuses


def draw_common_and_cf(old_draw_fn,old_rng,point_rng,natural_rng,cf_rng,pools,point_pools,natural_pools,cf_fit_count,use_cf):
    """CF RNG does not advance any common generator. All draws are with replacement."""
    draw=old_draw_fn(old_rng,pools)
    point=draw_balanced(point_rng,point_pools,16)
    natural=draw_balanced(natural_rng,natural_pools,8)
    if use_cf:need(type(cf_fit_count) is int and cf_fit_count>0,'nonempty constructed FIT parent pool; no fallback')
    cf=cf_rng.choice(cf_fit_count,4,replace=True) if use_cf else None
    return draw,point,natural,cf


def cf_parent_loss(logits):
    import torch.nn.functional as F
    need(logits.ndim==1 and logits.shape[0]==12,'four parents, ordered BEFORE/COVER/SHAM')
    z=logits.reshape(4,3)
    target=z.new_tensor([0.,1.,0.])[None].expand_as(z)
    bce=F.binary_cross_entropy_with_logits(z,target,reduction='none')
    return (bce*z.new_tensor([.25,.5,.25])[None]).sum(1).mean()


def matched_objective(out,point,natural,use_cf,five_losses):
    n=252 if use_cf else 240
    need(all(out[k].shape[0]==n for k in ('support_logits','owner_logits','direct_end_external_logit')),'exact old192/point32/natural16/(CF12) batch')
    common={k:out[k][:240] for k in ('support_logits','owner_logits','direct_end_external_logit')}
    loss,parts,strong,local=objective(common,point,natural,.25,five_losses)
    cf=cf_parent_loss(out['direct_end_external_logit'][240:252]) if use_cf else strong.new_zeros(())
    return loss+.25*cf,parts,strong,local,cf


def fixed_budget_from_probe(step_seconds_by_arm):
    """Root prospective rule, throughput only; no accuracy/loss argument exists."""
    need(set(step_seconds_by_arm)==set(ARMS),'both arm throughput measurements')
    values=list(step_seconds_by_arm.values())
    need(all(type(x) in (float,int) and math.isfinite(x) and x>0 for x in values),'finite positive step timing')
    return max(200,min(2000,100*math.floor((60/max(values))/100)))


def role_manifest(manifest,role,pairs,rows):
    need(manifest['completed'] is True and manifest['PASS'] is True and manifest['source_role']==role,'role-specific physical manifest')
    need(manifest['trainer_DEV_payloads_allowed_before_two_new_states_freeze'] is (role!='SOURCE_DEV'),'role-specific decoder boundary explicit')
    selected=[p for p in pairs if p['source_role']==role]
    success={p['parent_row_id'] for p in selected}
    need(set(manifest['successful_triplet_parent_ids'])==success,'same successful parent census')
    expected={r['row_id']:r for p in selected for r in p['rows']}
    supplied={r['row_id']:r for r in manifest['CF_variant_rows']}
    need(len(supplied)==len(manifest['CF_variant_rows']) and set(supplied)==set(expected),'all successful role-only triples')
    for rid,r in supplied.items():
        x=expected[rid]
        need(r['source_role']==role and same_binding(r['RGB96'],x['RGB96']) and r['valid96_RLE']==x['valid96_RLE'] and r['condition32']==x['condition32'],'manifest row image/condition identity')
    before={r['row_id']:r for r in rows if r['source_role']==role and r['variant']=='BEFORE'}
    bb={r['row_id']:r for r in manifest['all_parent_BEFORE_rows']}
    need(len(bb)==len(manifest['all_parent_BEFORE_rows']) and set(bb)==set(before),'failed factual BEFORE retained too')
    for rid,r in bb.items():need(same_binding(r['RGB96'],before[rid]['RGB96']),'factual BEFORE exact image binding')
    expected_images={(r['RGB96']['path'],r['RGB96']['sha256']) for r in list(expected.values())+list(before.values())}
    declared={(b['path'],b['sha256']) for b in manifest['exact_RGB96_whitelist']}
    need(declared==expected_images,'closed role-only image whitelist, not other role or source RGB')
    return manifest['exact_RGB96_whitelist']


def routed_objective(out,point,natural,separate_point_part,five_losses):
    """Same 252 rows/loss weights in both arms; only strong destination changes."""
    need(type(separate_point_part) is bool,'explicit prospective gradient routing')
    need(all(out[k].shape[0]==252 for k in ('support_logits','owner_logits','point_logit','part_logit')),'old192+point32+natural16+CF12 in both arms')
    base,parts=five_losses({k:out[k][:192] for k in ('support_logits','owner_logits')})
    strong_key='point_logit' if separate_point_part else 'part_logit'
    strong=balanced_loss(out[strong_key][192:224],point)
    local=balanced_loss(out['part_logit'][224:240],natural)
    cf=cf_parent_loss(out['part_logit'][240:252])
    return base+strong+.25*local+.25*cf,parts,strong,local,cf
