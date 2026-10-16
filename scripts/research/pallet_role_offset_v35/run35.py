from context35 import *
import subprocess

def main():
    stages=[('EXTRACTION_COMPLETE.json','extract35.py'),('ANALYSIS_COMPLETE.json','analyze35.py'),('FOLLOWUP_COMPLETE.json','followup35.py'),('QUANTIZATION.json','quantization35.py'),('AUDIT.json','audit35.py'),('GRID_AUDIT.json','audit_grid35.py'),('REPORT_COMPLETE.json','report35.py'),('COMPLETION.json','deliver35.py')]
    for marker,script in stages:
        if not (RUN/marker).exists():subprocess.run([sys.executable,str(KIT/script)],cwd=ROOT,check=True)
if __name__=='__main__':main()
