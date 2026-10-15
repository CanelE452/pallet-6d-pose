"""Independent likelihood/guard rebuild for new grazing mixture controls."""
from pathlib import Path
import argparse
import types

from . import audit_image_chain_mixture_control_v4_18 as AUDIT
from . import evaluate_grazing_image_chain_mixtures_v4_18b as B
from . import refine_grazing_image_chain_mixtures_v4_18c as CONTINUOUS


def main(generation):
    module=B if generation=='b'else CONTINUOUS
    facade=types.SimpleNamespace(**{**B.OLD.__dict__,**module.__dict__})
    runner=types.FunctionType(AUDIT.main.__code__,{**AUDIT.main.__globals__,'V':facade},AUDIT.main.__name__)
    runner(module.OUTPUT)
    C,E=facade.C,facade.E;output=module.OUTPUT;extra=[]
    if generation=='c':
        parents={r['id']:r for r in C.rows(B.OUTPUT/'PREDICTIONS.jsonl.gz')}
        for row in C.rows(output/'PREDICTIONS.jsonl.gz'):
            parent=parents[row['id']];native=row['arms'][module.ARMS[1]]
            previous=parent['arms'][B.ARMS[1]]
            if previous['accepted']:
                good=native['accepted']and native['solution']['objective']<=previous['solution']['objective']+1e-9
                extra.append(dict(id=row['id'],native_minimum_nonworsening=good))
                if not good:raise ValueError('Continuous pool omitted a feasible cached minimum')
    dest=output/'independent_root_audit_v1'
    C.write(dest/'WRAPPER_RECEIPT.json',dict(schema='GRAZING_MIXTURE_INDEPENDENT_REBUILD_WRAPPER_V18BC',
        generation=generation,source=E._binding(Path(__file__)),kernel_audit_source=E._binding(Path(AUDIT.__file__)),
        control_source=E._binding(Path(module.__file__)),audit=E._binding(dest/'AUDIT.json'),
        extra_checks=extra,extra_check_count=len(extra),new_fits_RGB_NN_or_reference_decodes=0))
    C.write(dest/'WRAPPER_SEAL.json',dict(receipt_sha256=C.sha(dest/'WRAPPER_RECEIPT.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('generation',choices=('b','c'))
    main(parser.parse_args().generation)
