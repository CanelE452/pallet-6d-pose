"""Audit exact image bytes against original YOLO training and all old refinement splits."""
from context21 import *
from cli_export import load_probe,load_population

def main():
    meta=load_probe();selected=read(RUN/'SOURCES.json')['rows'];own={r['merged_stem']:sha(ROOT/r['image_path']) for r in selected};assert len(set(own.values()))==506
    old={x for ids in load_population().values() for x in ids};forbidden=[r for r in meta.values() if r['dataset_split']=='train' or r['merged_stem'] in old];target=set(own.values());matches=[];bindings={}
    for i,r in enumerate(forbidden):
        h=sha(ROOT/r['image_path']);bindings[r['merged_stem']]=h
        if h in target:matches.append(r['merged_stem'])
        if (i+1)%10000==0:print('HASHED',i+1,len(forbidden),flush=True)
    assert not matches,matches
    write(RUN/'IMAGE_BINDINGS.json',dict(selected=own,forbidden=bindings));d=read(RUN/'SPLIT_AUDIT.json');d.update(status='PASS',exact_image_hash_overlap=0,forbidden_images_hashed=len(forbidden),unique_selected_image_hashes=506,bindings_sha256=sha(RUN/'IMAGE_BINDINGS.json'),limitation='Exact bytes and metadata pair groups checked; no guarantee of independent assets/scenes or exclusion from historical YOLO validation selection.');write(RUN/'SPLIT_AUDIT.json',d);print('SPLIT HASH PASS',flush=True)

if __name__=='__main__':main()
