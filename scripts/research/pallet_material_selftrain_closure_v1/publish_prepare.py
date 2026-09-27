"""Prepare only the user-authorized publication payload, without Git/network."""
from . import common as C

EXCLUDED=['POOL.json','WOOD_DATA_INVENTORY.json','PREFLIGHT.json','WOOD_TRAIN_PROTOCOL.json',
          'FIT_WOOD_RAW_LR5.json','FIT_WOOD_REF_LR5.json']

def main():
    assert C.read(C.DOC/'PUBLICATION_AUTHORIZATION.json')['status']=='USER_AUTHORIZED'
    for arm in ('WOOD_RAW_LR5','WOOD_REF_LR5'):
        original=C.DOC/f'FIT_{arm}.json';fit=C.read(original)
        for event in fit['history']:
            for key in ('compute','foreign_compute','rustdesk_preserved'):
                event.get('gpu',{}).pop(key,None)
        fit['private_original']=C.bind(original)
        fit['redaction']='Unrelated process PID/executable details omitted; checkpoint/training/protected-state metadata unchanged.'
        C.save(C.DOC/f'FIT_{arm}_PUBLIC.json',fit)
    state=C.read(C.DOC/'EXECUTION_STATUS.json')
    state.update(status='EXPERIMENT_PAPER_AUDIT_COMPLETE_USER_AUTHORIZED_PUSH',remaining_blocker=None,
                 approval_received=True,publication_authorization=C.bind(C.DOC/'PUBLICATION_AUTHORIZATION.json'))
    state['latest_local_review'].update(approval_received=True)
    C.save(C.DOC/'EXECUTION_STATUS.json',state)
    docs=[p for p in sorted(C.DOC.iterdir()) if p.is_file() and p.suffix in ('.json','.md','.tex') and p.name not in EXCLUDED]
    forbidden={'K','keypoints_xy','gt','verified_xy','model_state_dict'}
    def check(value):
        if isinstance(value,dict):
            assert not set(value)&forbidden,set(value)&forbidden
            for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
    for p in docs:
        if p.suffix=='.json':check(C.read(p))
    C.save(C.DOC/'PUBLICATION_MANIFEST.json',dict(
        authorized=True,docs=[str(p.relative_to(C.ROOT)) for p in docs],
        excluded_local_documents=EXCLUDED,
        figures=[str(p.relative_to(C.ROOT)) for p in sorted((C.DOC/'figures').glob('*.png'))],
        original_RGB=False,raw_coordinates=False,camera_matrices=False,weight_files=False,
        approval=C.bind(C.DOC/'PUBLICATION_AUTHORIZATION.json')))
    print('AUTHORIZED_PUBLICATION_READY',len(docs),'documents',flush=True)

if __name__=='__main__':main()
