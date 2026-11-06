"""One aggregate-only paired-seed figure; no model or article data access."""
from pathlib import Path
import os
import tempfile
os.environ['MPLCONFIGDIR']=str(Path(tempfile.gettempdir())/'distilroberta-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=pd.read_csv(ROOT/'outputs/tables/distilroberta_budget_seed_results.csv')
    rows=rows[rows.scope.eq('same_target_holdout')]
    if len(rows)!=6:raise ValueError('Expected six completed seed-budget evaluations')
    fig,axes=plt.subplots(1,3,figsize=(11,4.4),sharey=True,layout='constrained')
    for ax,metric,label in zip(axes,['macro_f1','roc_auc','average_precision'],['Macro F1','ROC-AUC','Average precision']):
        for seed,color in zip([42,123,2026],['#2563a6','#cc7225','#37835b']):
            part=rows[rows.seed.eq(seed)].set_index('max_length')
            ax.plot([0,1],[part.loc[512,metric],part.loc[256,metric]],marker='o',lw=2,color=color,label=f'Seed {seed}')
        ax.set(xticks=[0,1],xticklabels=['512 tokens','256 tokens'],xlim=(-.2,1.2),ylim=(0,1),title=label)
        ax.spines[['top','right']].set_visible(False);ax.yaxis.grid(alpha=.2)
    axes[0].set_ylabel('Target holdout score')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,frameon=False)
    fig.suptitle('FakeHealth → CoAID: paired source-token budgets across seeds\nIdentical unpadded target content; fixed source hyperparameters')
    path=ROOT/'outputs/figures/distilroberta/fh_to_coaid_budget_seed_paired_deltas.svg'
    fig.savefig(path,bbox_inches='tight',metadata={'Creator':f'Matplotlib {matplotlib.__version__}'})
    fig.savefig(Path(tempfile.gettempdir())/'fh_to_coaid_budget_seed_paired_deltas.png',dpi=140,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
