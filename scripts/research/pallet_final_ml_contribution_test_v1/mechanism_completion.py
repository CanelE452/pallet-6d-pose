"""Complete optional domain metadata using the unchanged canonical CSV."""
import csv
from common import *
from statistics_and_mechanism import load,mechanism

def run():
    stores,poses,preds,targets,metadata,summaries=load()
    path=old('paper_evaluation').OLDCSV
    with path.open() as f:
        canonical={r['frame_id']:r['domain'] for r in csv.DictReader(f) if r['kind']=='POSITIVE'}
    patched=[]
    for frame,item in metadata.items():
        if 'domain' not in item:
            item['domain']=canonical[frame] or 'UNTAGGED'
            patched.append(dict(frame_id=frame,domain=item['domain']))
        else:assert item['domain']==canonical[frame],frame
    write(B/'MECHANISM_METADATA_ADAPTER.json',dict(
        source_manifest_sha256=sha(old('paper_evaluation').POS),canonical_csv_sha256=sha(path),
        code_sha256=sha(Path(__file__)),filled=patched,
        rule='Missing optional manifest domain is filled from its existing canonical paper CSV; blank remains UNTAGGED. No outcome/GT-error classification.',
        locked_scoring_source_unchanged=True,predictions_and_statistics_unchanged=True))
    mechanism(preds,targets,metadata,stores)
    print('MECHANISM_COMPLETE',len(patched),'missing manifest domains supplied from canonical CSV',flush=True)

if __name__=='__main__':run()
