from common import *
from evaluate import ci
import numpy as np
def main():
    folder=OUT/'foreground_matched';s=read(folder/'SUMMARY.json');f=read(folder/'PER_FRAME.json');rows=read(folder/'test_records.json');out={}
    for pop,seen in [('seen',True),('unseen',False)]:
        ix=np.array([r['seen']==seen for r in rows]);groups=[r['group'] for r in rows if r['seen']==seen]
        c=np.array([[r['normalized'] for r in f[f'selected_s{i}']] for i in [1,2,3]])[:,ix]
        a=np.array([[r['normalized'] for r in f[f'direct_s{i}']] for i in [1,2,3]])[:,ix];b=np.array([r['normalized'] for r in f['stock']])[ix]
        mc=np.mean([s['results'][f'selected_s{i}'][pop]['overall']['matched_mean_px'] for i in [1,2,3]])
        ma=np.mean([s['results'][f'direct_s{i}'][pop]['overall']['matched_mean_px'] for i in [1,2,3]]);mb=s['results']['stock'][pop]['overall']['matched_mean_px']
        out[pop]=dict(stock=mb,direct=ma,selected=mc,selected_matched_relative_gain=1-mc/mb,direct_matched_relative_gain=1-ma/mb,
            selected_vs_stock_ci=ci(c.mean(0)-b,groups),selected_vs_direct_ci=ci((c-a).mean(0),groups),
            selected_vs_stock_primary_gain_per_seed=(1-c.mean(1)/b.mean()).tolist())
    write(folder/'CONDITIONAL_ANALYSIS.json',dict(posthoc=True,changes_primary_decision=False,populations=out))
if __name__=='__main__':main()
