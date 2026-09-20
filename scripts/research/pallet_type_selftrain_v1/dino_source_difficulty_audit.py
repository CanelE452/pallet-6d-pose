"""Read-only source supervision audit; never changes samples, masks or labels."""
from collections import Counter,defaultdict
import numpy as np
from . import dino_mid_feature as U


def main():
    p=U.verify();parent=U.X.verify();assert p['source_samples']==parent['source_samples']
    exposures=Counter(np.asarray(p['source_samples']).ravel().tolist())
    rows=[];summary={};groups=defaultdict(list)
    for r in U.C.read(U.X.DOC/'CACHE_COMPLETE.json')['records']:
        if r['domain']!='source':continue
        with np.load(U.C.ROOT/r['cache']['path']) as z:
            q=np.array(z['points'])[:8];valid=np.array(z['valid'])[:8];target=np.array(z['target'])[:8]
            mask=np.array(z['target_valid'])[:8];gain=float(z['matrix'][0,0])
        valid&=np.isfinite(q).all(-1);assert valid.any() and gain>0
        error=np.linalg.norm(q-target,axis=-1)/gain
        nearest=np.linalg.norm(target[:,None,:]-q[valid][None,:,:],axis=-1).min(-1)/gain
        group='train' if r['train'] else 'held'
        item=dict(row=r['row'],id=r['id'],partition=group,corners=int(mask.sum()),
            good5=int(((error<5)&mask).sum()),hard20=int(((error>20)&mask).sum()),
            far40=int(((nearest>40)&mask).sum()),exposures=exposures.get(r['row'],0))
        assert item['exposures']>0 if r['train'] else item['exposures']==0
        rows.append(item);groups[group].append(item)
    for group,rr in groups.items():
        n=sum(r['corners'] for r in rr)
        summary[group]=dict(images=len(rr),corners=n,images_with_far40=sum(r['far40']>0 for r in rr),
            good5=sum(r['good5'] for r in rr),hard20=sum(r['hard20'] for r in rr),far40=sum(r['far40'] for r in rr),
            hard20_fraction=sum(r['hard20'] for r in rr)/n,far40_fraction=sum(r['far40'] for r in rr)/n,
            scheduled_image_exposures=sum(r['exposures'] for r in rr),
            scheduled_corner_exposures=sum(r['exposures']*r['corners'] for r in rr),
            scheduled_far40_exposures=sum(r['exposures']*r['far40'] for r in rr))
    assert summary['train']['images']==8192 and summary['held']['images']==64
    assert summary['train']['scheduled_image_exposures']==40000
    U.C.freeze(U.DOC/'SOURCE_DIFFICULTY_AUDIT.json',dict(status='SOURCE_ONLY_DIAGNOSTIC_NO_PROTOCOL_CHANGE',summary=summary,rows=rows,
        definition='Fixed source-native GT identities as used by loss; distances in prepared source RGB px. Far40 means no valid existing R0 corner within40px of the supervised target. No real GT or predicted corrected coordinates used.',
        exposure_not_loss_weight='Counts are exposures,not exact CE contribution fractions; each microbatch uses its own supported-corner denominator.',
        predictions_changed=False,samples_changed=False,targets_changed=False,
        evidence=[U.C.bound(__file__),U.C.bound(U.DOC/'PROTOCOL.json'),U.C.bound(U.X.DOC/'CACHE_COMPLETE.json')]))
    print('SOURCE_DIFFICULTY_AUDIT',summary,flush=True)


if __name__=='__main__':main()
