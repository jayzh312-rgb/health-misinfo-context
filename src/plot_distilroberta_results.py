"""Render only saved aggregate results; no model or raw-text access.

Uses existing Matplotlib read-only; set PYTHONDONTWRITEBYTECODE=1.
"""
from pathlib import Path
import json
import os
import tempfile
os.environ['MPLCONFIGDIR']=str(Path(tempfile.gettempdir())/'distilroberta-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'outputs/tables'
OUT=ROOT/'outputs/figures/distilroberta'
COLORS=['#64748b','#d18b2c','#2563a6']


def save(fig,name):
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',metadata={'Creator':f'Matplotlib {matplotlib.__version__}'})
    fig.savefig(Path(tempfile.gettempdir())/f'distilroberta_{name}.png',dpi=140,bbox_inches='tight')
    plt.close(fig)


def grouped(values,labels,series,title,name):
    fig,ax=plt.subplots(figsize=(max(7,len(labels)*2.2),4.6),layout='constrained')
    x=np.arange(len(labels));width=.8/len(series)
    for j,label in enumerate(series):
        bars=ax.bar(x-.4+width*(j+.5),values[:,j],width*.9,label=label,color=COLORS[j])
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.07),ylabel='Macro F1',title=title)
    ax.spines[['top','right']].set_visible(False)
    ax.yaxis.grid(True,alpha=.2);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=len(series),frameon=False)
    save(fig,name)


def main():
    compare=pd.read_csv(TABLES/'distilroberta_vs_classical.csv')
    short={'FakeHealth':'FH','CoAID':'CA'}
    for evaluation in ['within','transfer']:
        d=compare[compare.evaluation.eq(evaluation)&compare.metric.eq('macro_f1')]
        labels=[r.target_dataset if evaluation=='within' else
            r.direction.replace('_to_',' → ')+'\n'+('All eligible' if r.scope=='all_eligible' else 'Target holdout') for r in d.itertuples()]
        grouped(d[['logistic','linear_svm','distilroberta']].to_numpy(),labels,
            ['Logistic regression','Linear SVM','DistilRoBERTa (512)'],
            'Within-context holdout performance' if evaluation=='within' else 'Cross-context generalization',
            evaluation+'_model_comparison')
    transfers=pd.read_csv(TABLES/'distilroberta_transfer_results.csv')
    primary=transfers[transfers.max_length.eq(512)]
    fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained')
    vmax=primary[['tn','fp','fn','tp']].to_numpy().max()
    for ax,r in zip(axes.ravel(),primary.itertuples()):
        matrix=np.array([[r.tn,r.fp],[r.fn,r.tp]])
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=vmax)
        for (i,j),v in np.ndenumerate(matrix):
            ax.text(j,i,str(v),ha='center',va='center',fontsize=15,color='white' if v>vmax*.55 else '#172536')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],
            xlabel='Predicted',ylabel='Actual',
            title=r.direction.replace('_to_',' → ')+' · '+('all eligible' if r.scope=='all_eligible' else 'target holdout')+f'\nN = {r.N:,}')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label='Article count')
    fig.suptitle('DistilRoBERTa transfer confusion matrices · 512 tokens · threshold 0.5')
    save(fig,'transfer_confusion_matrices')
    within=pd.read_csv(TABLES/'distilroberta_holdout_results.csv')
    data=pd.concat([within,transfers[transfers.scope.eq('same_target_holdout')]],ignore_index=True)
    p=data[data.max_length.eq(512)];s=data[data.max_length.eq(256)].set_index('direction')
    labels=['Within '+short[r.target_dataset] if r.evaluation=='within' else r.direction.replace('_to_',' → ') for r in p.itertuples()]
    vals=np.array([[r.macro_f1,s.loc[r.direction,'macro_f1']] for r in p.itertuples()])
    grouped(vals,labels,['512 tokens','256 tokens'],
        'Token-budget sensitivity on identical holdout rows\n256 reuses the 512-selected hyperparameters',
        '512_vs_256_sensitivity')
    with (ROOT/'logs/distilroberta_text_experiment.log').open('a') as f:
        f.write('Saved four aggregate-only figures using Matplotlib '+matplotlib.__version__+'; no models or raw text read.\n')


if __name__=='__main__':main()
