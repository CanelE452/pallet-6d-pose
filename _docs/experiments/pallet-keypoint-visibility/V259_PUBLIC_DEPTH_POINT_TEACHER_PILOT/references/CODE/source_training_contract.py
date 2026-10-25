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
