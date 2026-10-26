from common import *
import argparse
def main():
    a=argparse.ArgumentParser();a.add_argument('phase',choices=('before','after'));args=a.parse_args()
    start('integrity_'+args.phase)
    result={'phase':args.phase,'utc':now(),'mode':'opaque byte SHA256 only; no annotation/image decoding','roots':{},'R0':{'path':str(R0),'sha256':sha(R0),'bytes':R0.stat().st_size}}
    for root in (FLIP,NOFLIP):
        values={}
        for i,p in enumerate(sorted(root.rglob('*'))):
            if p.is_file():
                values[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size}
        result['roots'][str(root)]=values
        print(args.phase,root.name,'files',len(values),'bytes',sum(v['bytes'] for v in values.values()),flush=True)
    dest=OUT/'hashes'/('SNAPSHOT_'+args.phase.upper()+'.json');write(dest,result)
    if args.phase=='after':
        before=read(OUT/'hashes/SNAPSHOT_BEFORE.json');diff=[]
        for root,old in before['roots'].items():
            new=result['roots'][root]
            for rel in sorted(set(old)|set(new)):
                if old.get(rel)!=new.get(rel):diff.append({'root':root,'path':rel,'before':old.get(rel),'after':new.get(rel)})
        if before['R0']!=result['R0']:diff.append({'path':str(R0),'before':before['R0'],'after':result['R0']})
        write(OUT/'hashes/COMPARISON.json',{'all_unchanged':not diff,'differences':diff,'before_sha256':sha(OUT/'hashes/SNAPSHOT_BEFORE.json'),'after_sha256':sha(dest)})
        assert not diff,diff
    log('step_complete',task='integrity_'+args.phase,output=str(dest),output_sha256=sha(dest))
if __name__=='__main__':main()
