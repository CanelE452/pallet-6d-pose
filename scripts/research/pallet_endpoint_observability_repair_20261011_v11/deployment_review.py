"""Compatibility repair for repeated, identical frame fields in V2 packet.

The frozen geometry/solver stays unchanged. Assemble_view produces a complete
frame row; V2 packet expects only the result portion. Verify and remove just
its repeated metadata fields before that unchanged packet function.
"""
from .deployment import Pipeline as Original
from ..pallet_sparse_local_line_20261010_v9 import common as C

FIELDS=('id','session','method','K','xyz','raw_hw','selected_index','fixed_metadata')


class Pipeline(Original):
    def __init__(self,args):
        super().__init__(args)
        original=self.base.packet
        self.packet_compatibility_counts={'calls':0,'identical_fields_removed':0}
        def packet(meta,capture,method,result):
            expected=original(meta,capture,method,{})
            payload=dict(result);removed=[]
            for field in FIELDS:
                if field in payload:
                    C.require(C.semantic_sha256(payload[field])==C.semantic_sha256(expected[field]),
                              'packet compatibility differs: '+field)
                    del payload[field];removed.append(field)
            value=original(meta,capture,method,payload)
            self.packet_compatibility_counts['calls']+=1
            self.packet_compatibility_counts['identical_fields_removed']+=len(removed)
            value['packet_compatibility_review']=dict(removed_identical_fields=removed,
                numerical_result_changed=False,original_frame_metadata_preserved=True,
                original_packet_function_used=True)
            return value
        self.base.packet=packet

