"""Small immutable IO and measured accounting for fixed-pose analysis."""
from contextlib import contextmanager
import csv
import gzip
import io
import json
import os
from pathlib import Path
import time

from scripts.research.pallet_statewise_correspondence_diagnosis_v1.common import (
    canonical, digest, finite, read, rows, save_rows, sha, write,
)


def write_csv(path, values, fields=None):
    values=iter(values)
    first=next(values,None)
    if fields is None:
        fields=list(first) if first is not None else []
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    raw=p.open('xb')
    binary=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) if p.suffix=='.gz' else raw
    with raw:
        with io.TextIOWrapper(binary,encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
            writer.writeheader()
            def emit(row):
                writer.writerow({k:('NA' if row.get(k) is None else
                    canonical(row[k]) if isinstance(row[k],(dict,list,tuple)) else row[k])
                    for k in fields})
            if first is not None:emit(first)
            for row in values:emit(row)


def csv_rows(path):
    p=Path(path)
    with (gzip.open(p,'rt',newline='') if p.suffix=='.gz' else p.open(newline='')) as stream:
        yield from csv.DictReader(stream)


@contextmanager
def timed_stage(folder,name,**details):
    started,cpu=time.monotonic(),time.process_time()
    receipt={}
    try:
        yield receipt
    finally:
        receipt.update(stage=name,elapsed_seconds=time.monotonic()-started,
                       process_cpu_seconds=time.process_time()-cpu,**details)
        p=Path(folder)/'RESOURCE_LEDGER.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
        data=(canonical(receipt)+'\n').encode()
        fd=os.open(p,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
        try:os.write(fd,data)
        finally:os.close(fd)
