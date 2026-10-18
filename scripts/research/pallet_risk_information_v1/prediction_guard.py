"""Child-process Python I/O guard for prediction, independent of file naming.

This is auditable Python I/O instrumentation, not an operating-system sandbox.
Use a fresh prediction process; do not preload references or pass reference objects.
"""
from __future__ import annotations
import os,sys
from pathlib import Path


class PredictionGuard:
    def __init__(self, *, allowed_files=(), allowed_read_roots=(), output_root=None,
                 denied_files=(), denied_roots=()):
        self.allowed_files={Path(p).resolve()for p in allowed_files}
        self.allowed_read_roots=tuple(Path(p).resolve()for p in allowed_read_roots)
        self.denied_files={Path(p).resolve()for p in denied_files}
        self.denied_roots=tuple(Path(p).resolve()for p in denied_roots)
        self.output_root=Path(output_root).resolve()if output_root else None
        self.denied_inodes=set()
        for p in self.denied_files:
            if p.exists():s=p.stat();self.denied_inodes.add((s.st_dev,s.st_ino))
        self.active=False;self.events=[];self.blocked=[];self._resolving=False

    @staticmethod
    def under(path,root):return path==root or path.is_relative_to(root)

    def _deny(self,event,path,reason):
        item=dict(event=event,path=str(path),reason=reason);self.blocked.append(item)
        raise PermissionError('Prediction I/O guard rejected '+event+': '+reason)

    def hook(self,event,args):
        if not self.active or self._resolving:return
        if event in('subprocess.Popen','os.system','os.posix_spawn','os.posix_spawnp','os.exec','os.fork','socket.connect','socket.connect_ex'):
            self._deny(event,'', 'subprocess/network escape is outside prediction I/O contract')
        if event!='open':return
        value,mode,flags=args
        if isinstance(value,int):self._deny(event,'<fd>', 'unregistered inherited file descriptor')
        try:
            self._resolving=True;path=Path(os.fsdecode(value)).resolve()
            inode=None
            if path.exists():s=path.stat();inode=(s.st_dev,s.st_ino)
        finally:self._resolving=False
        if path in self.denied_files or any(self.under(path,r)for r in self.denied_roots)or inode in self.denied_inodes:
            self._deny(event,path,'reference/evaluation input or its filesystem alias')
        writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC))
        if writing:
            if self.output_root is None or not self.under(path,self.output_root):self._deny(event,path,'write outside new prediction output')
        elif path not in self.allowed_files and not any(self.under(path,r)for r in self.allowed_read_roots):
            self._deny(event,path,'read is absent from explicit predictor whitelist')
        self.events.append(dict(event=event,path=str(path),mode='write'if writing else'read'))

    def __enter__(self):
        sys.addaudithook(self.hook);self.active=True;return self

    def __exit__(self,*_):self.active=False

    def receipt(self):
        return dict(schema='REFERENCE_FREE_PREDICTION_IO_GUARD_V1',allowed_files=sorted(map(str,self.allowed_files)),
            allowed_read_roots=list(map(str,self.allowed_read_roots)),output_root=str(self.output_root)if self.output_root else None,
            denied_files=sorted(map(str,self.denied_files)),denied_roots=list(map(str,self.denied_roots)),
            observed_open_count=len(self.events),blocked_attempt_count=len(self.blocked),events=self.events,blocked=self.blocked,
            enforcement='Fresh child-process Python audit hook; resolved path+explicit files/roots+denied inode aliases; subprocess/network escape denied',
            limitation='Not an OS sandbox; cannot inspect already loaded objects, uninstrumented native file I/O or prior inherited descriptors. Prediction must run fresh and accept only whitelisted feature/checkpoint inputs.')
