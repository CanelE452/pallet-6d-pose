"""Bind completed control analysis to the predeclared descriptive rules."""
from pathlib import Path
from . import common as C
from .controls import contrast_pairs
from .robustness import run


def main():
    control=C.read(C.DOC/'CONTROL_RESULTS.json')
    for binding in control['sources']+control['private_artifacts']:
        C.verify(binding)
    path=C.DOC/'ROBUSTNESS_PAIRS.json'
    C.save(path,contrast_pairs(),True)
    run(C.RAW/'CONTROL_FRAME_METRICS_PRIVATE.json',
        C.OLD.RAW/'evaluation/S42/METADATA.json',path,'CONTROL_ROBUSTNESS.json')


if __name__=='__main__':
    main()
