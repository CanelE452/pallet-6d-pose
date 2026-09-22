"""Fail-fast subprocess sequence; never starts another fit after a failure."""
import subprocess
import sys
from . import common as C


def main():
    assert C.read(C.DOC/'LOSS_TEST.json')['passed']
    assert C.read(C.DOC/'PROBE_BEFORE.json')['complete']
    commands=[]
    for mat in C.MATERIALS:
        for arm in C.ARMS:commands.append(('train',[mat,arm]))
    for mat in C.MATERIALS:
        for arm in C.ARMS:commands.append(('evaluate',['infer','--material',mat,'--arm',arm]))
    commands.append(('evaluate',['score']))
    for number,(module,args) in enumerate(commands):
        path=C.RAW/'logs'/f'{number:02d}_{module}_{"_".join(args)}.log';path.parent.mkdir(parents=True,exist_ok=True)
        print('START',module,args,flush=True)
        with path.open('x') as log:
            result=subprocess.run([sys.executable,'-m','scripts.research.pallet_clean19_structured_easyhard_v1.'+module,*args],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            C.save(C.DOC/'STOP.json',dict(stage=module,args=args,exit_code=result.returncode,log=C.bind(path),automatic_retry=False))
            raise SystemExit(result.returncode)
        print('DONE',module,args,flush=True)
    C.save(C.DOC/'DRIVER_COMPLETE.json',dict(complete=True,fits=6,optimizer_updates=1920))


if __name__=='__main__':main()
