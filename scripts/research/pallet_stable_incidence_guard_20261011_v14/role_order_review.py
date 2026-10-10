"""Normalize the preserved audit's tuple policy to its JSON representation."""
import json
from pathlib import Path
from . import role_order_audit as K

FILES=('REVIEW_ROLE_ORDER_AUDIT_PROTOCOL.json','REVIEW_ROLE_ORDER_AUDIT_STARTED.json',
    'REVIEW_ROLE_ORDER_AUDIT_CHECKS.json','REVIEW_ROLE_ORDER_AUDIT_ROWS.jsonl.gz')

def main():
    protocol=K.read(K.DOC/K.FILES[0]);old_inputs,old_authority,old_write,old_files=K.inputs,K.authority,K.write,K.FILES
    K.require(protocol['inputs']==old_inputs() and protocol['policy']==json.loads(json.dumps(old_authority())),
        'preserved original tuple/list-only serialization mismatch')
    failed=K.read(K.DOC/'ROLE_ORDER_AUDIT_FIRST_FAILURE.json')
    K.require(failed['comparisons_started']==0 and failed['failure']=='own fixed audit differs'
        and failed['original_protocol']==K.binding(K.DOC/old_files[0]),'preserved before-arithmetic failure')
    def inputs():
        b=old_inputs()
        for key,p in (('review',Path(__file__).resolve()),('original_protocol',K.DOC/old_files[0]),
            ('original_failure',K.DOC/'ROLE_ORDER_AUDIT_FIRST_FAILURE.json')):b[key]=K.binding(p)
        return b
    def authority():return json.loads(json.dumps(old_authority()))
    def write(path,value):
        value['serialization_review']=dict(original_failed_attempt_preserved=True,
            tuple_list_JSON_equality_setup_only=True,numerical_comparisons_or_pose_policy_changed=False,
            physical_order_is_annotation_convention_not_universal_all_rotations=True,
            baseline_applies_soft_order_penalty_not_hard_universal_pose_rule=True)
        old_write(path,value)
    try:
        K.inputs,K.authority,K.write,K.FILES=inputs,authority,write,FILES;K.main()
    finally:K.inputs,K.authority,K.write,K.FILES=old_inputs,old_authority,old_write,old_files
if __name__=='__main__':main()
