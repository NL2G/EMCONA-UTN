import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from glob import glob
from collections import Counter, defaultdict
from scipy.stats import pearsonr, kendalltau, spearmanr, ttest_ind
from sklearn.metrics import classification_report, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('darkgrid')

def min_max_norm(X):
    X = np.array(X)
    X_min = np.min(X)
    X_max = np.max(X)
    return (X - X_min) / (X_max - X_min)

def z_score_norm(X):
    X_i = [x for x in X if x != 100] # remove N/A
    m = np.mean(X_i)
    std = np.std(X_i)
    X = [(x-m)/std if x != 100 else 100 for x in X]
    return X


def process_row(x, xo, order=1):
    # quiz 
    ans = {
        0: "Circular Claim",
        1: "Ad Populum", 
        2: "Faulty Generalization",
        3: "False Causality", 
        4: "Ad Hominem"
    }
    # 8: 
    annotators = []
    anno_score = {}
    
    for j, row in x.iterrows():
        # only consider the first three received annotations
        if j > 2:
            break
        annotator = row['Please enter your Prolific ID.']
        answers = row.values[3:8]
        score = [ans[i] == a for i, a in enumerate(answers)]
        anno_score[annotator] = score
        annotations = row.values[8:]
        if order == 2:
            emo_labels = [a.lower() for i, a in enumerate(annotations) if i%3==2]
            conv_labels = [int(a.split()[0]) if isinstance(a, str) else 100 for i, a in enumerate(annotations) if i%3==0]
            fallacy_labels = [a.lower() for i, a in enumerate(annotations) if i%3==1]
        elif order == 1:
            emo_labels = [a.lower() for i, a in enumerate(annotations) if i%3==0]
            conv_labels = [int(a.split()[0]) if isinstance(a, str) else 100 for i, a in enumerate(annotations) if i%3==2]
            fallacy_labels = [a.lower() for i, a in enumerate(annotations) if i%3==1]
        else:
            raise NotImplementedError
        assert len(emo_labels) == len(conv_labels) == len(fallacy_labels)
        
        annotators.append(annotator)
        xo[f'emo_{j}'] = emo_labels
        xo[f'fallacy_{j}'] = fallacy_labels
        xo[f'conv_{j}'] = conv_labels
        xo[f'conv_zscore_{j}'] = z_score_norm(conv_labels)
    meta = {}

    anno_agree = defaultdict(lambda: defaultdict(list))
    full_agree = defaultdict(lambda: defaultdict(list))
    maj_agree = defaultdict(lambda: defaultdict(list))
    for l in ['emo', 'fallacy', 'conv', 'conv_zscore']:
        xo[l] = [[row[f"{l}_{j}"] for j in range(3)] for _, row in xo.iterrows()]
        values = []
        if l not in ['conv_score']:
            full_agree[l] = sum([row[f"{l}_0"] == row[f"{l}_1"] and row[f"{l}_2"] == row[f"{l}_1"] for _, row in xo.iterrows()]) / len(xo)
            maj_agree[l] = sum([len(set([row[f"{l}_0"],row[f"{l}_1"], row[f"{l}_2"]])) <= 2 for _, row in xo.iterrows()]) / len(xo)
        for i in range(3):
            for j in range(i+1, 3):
                if l in ['conv', 'conv_zscore']:
                    l1 = [l1 for l1, l2 in zip(xo[f"{l}_{i}"].tolist(), xo[f"{l}_{j}"].tolist()) if l1!=100 and l2!=100]
                    l2 = [l2 for l1, l2 in zip(xo[f"{l}_{i}"].tolist(), xo[f"{l}_{j}"].tolist()) if l1!=100 and l2!=100]
                    correlation = pearsonr(l1, l2)[0] 
                    values.append(correlation)
                    anno_agree[l][i].append(correlation)
                    anno_agree[l][j].append(correlation)
                else:
                    kappa = cohen_kappa_score(xo[f"{l}_{i}"].tolist(), xo[f"{l}_{j}"].tolist())
                    values.append(kappa)
                    anno_agree[l][i].append(kappa)
                    anno_agree[l][j].append(kappa)
                
        meta[l] = np.mean(values)
    best = {}
    for label, annot in anno_agree.items():
        for a, v in annot.items():
            anno_agree[label][a] = np.mean(v) 
        best[label] = sorted(anno_agree[label].items(), key=lambda x: x[1], reverse=True)[0][0]
        xo[f"{label}_best_annotator"] = [best[label]] * len(xo)
    
    return xo, meta, anno_score, full_agree, maj_agree


'''
# mask prolific id
paths = sorted(glob("outputs_all/annotation/batch_*（回复）.xlsx"))

ids = 0
for path in paths:
    df = pd.read_excel(path)
    df['Please enter your Prolific ID.'] = range(ids, ids+len(df))
    ids += len(df)
    print(df['Please enter your Prolific ID.'])
    file = path.split("/")[-1]
    df.to_excel(f"../argument/EMCONA-UTN/emotion_fallacy/data/annotations/{file}", index=False)

raise ValueError
'''

paths = sorted(glob("data/annotations/batch_*（回复）.xlsx"))
df = []

anno_score = {}
agreements = defaultdict(list)

batch_annotator_eval = defaultdict(dict)
for path in paths:
    print(path)
    batch = int(path.split("_")[-1].split("（")[0])
    tmp = pd.read_excel(path)
    tmp.astype(dtype= {"How convincing do you find this argument?": str})
    tmp_o = pd.read_csv(f"data/forms/batch_{batch}.tsv", sep='\t')

    # 
    if (batch > 5 and batch < 11) or (batch > 15):
        tmo_o, agreement, anno_eval, full_agree, maj_agree = process_row(tmp, tmp_o, order=2)
    else:
        tmo_o, agreement, anno_eval, full_agree, maj_agree = process_row(tmp, tmp_o, order=1)
    anno_score.update(anno_eval)
    for i, (k,v) in enumerate(anno_eval.items()):
        batch_annotator_eval[batch][i] = sum(v)
    
    agreements['batch'].append(batch)
    for k, v in agreement.items():
        if k not in ['conv_zscore']:
            agreements[k+'_full'].append(full_agree[k])
            agreements[k+'_maj'].append(maj_agree[k])
        agreements[k].append(v)
        
    tmp_o['batch'] = [batch] * len(tmp_o)
    df.append(tmp_o)
    
df = pd.concat(df, ignore_index=True)

print("all annotations:")
print(df)
df.to_csv("data/annotations/merged.csv", index=False)
agreements = pd.DataFrame(agreements)
print("full agreement results:")
print(agreements)
agreements.to_csv("data/results/agreement.csv", index=False)


print(f"average quiz grade - {len(anno_score)} annotators:", np.mean([sum(v) for v in list(anno_score.values())]))


sample = pd.read_csv("data/sampled_1000.tsv", sep='\t') # get gold labels / meta data for arguments

df['strategy_gen'] = df['id'].apply(lambda x: 'N/A' if x==-1 else sample[sample.id==x].iloc[0]['method'])
df['emotion_gen'] = df['id'].apply(lambda x: 'N/A' if x==-1 else sample[sample.id==x].iloc[0]['emotion'])
df['model_gen'] = df['id'].apply(lambda x: 'N/A' if x==-1 else sample[sample.id==x].iloc[0]['model'])
df['fallacy_gold'] = df['id_ori'].apply(lambda x: sample[sample.id_ori==x].iloc[0]['fallacy'])
df['fallacy_gold'] = df['fallacy_gold'].apply(lambda x: x if 'circular' not in x else 'circular claim') # label circular reasoning --> circular claim

# individual

r = defaultdict(lambda: defaultdict(list))
r = defaultdict(list)
for _, row in df.iterrows():
    for label in ['emo', 'conv', 'fallacy', 'conv_zscore', 'fallacy_gold', 'model_gen']:
        if 'gold' not in label and 'target' not in label and 'gen' not in label:
            r[label] += [row[f"{label}_{i}"] for i in range(3)]
        else:
            r[label] += [row[label]] * 3

# 3000 annotations = 1000 arguments x 3 annotators
r = pd.DataFrame(r)

# label distribution - Figure 2
plt.figure(figsize=(3,2.5))
ax = sns.histplot(data=r[r.conv!=100], y='conv', binwidth=0.8, stat='percent')
ax.set_ylabel("Convincingness")
plt.tight_layout()
plt.savefig("data/results/conv_dist.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(3.5,2))
ax = sns.histplot(data=r, y='emo', stat='percent')
ax.set_ylabel("Emotion")
plt.tight_layout()
plt.savefig("data/results/emo_dist.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(3.5,2))
ax = sns.histplot(data=r.sort_values('fallacy'), y='fallacy', stat='percent')
ax.set_ylabel("Fallacy")
plt.tight_layout()
plt.savefig("data/results/fallacy_dist.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# human perceived fallacious vs. fallacy-free argument distribution - Figure 3
rr = defaultdict(list)
r_syn = r[r.model_gen!='N/A']
r_ori = r[r.model_gen=='N/A']

rr['Argument'].append('original')
rr['Label'].append('fallacy-free')
rr['Percent'].append(len(r_ori[r_ori.fallacy=='none'])/len(r_ori) * 100)

rr['Argument'].append('original')
rr['Label'].append('fallacious')
rr['Percent'].append(len(r_ori[r_ori.fallacy!='none'])/len(r_ori) * 100)


rr['Argument'].append('synthetic')
rr['Label'].append('fallacy-free')
rr['Percent'].append(len(r_syn[r_syn.fallacy=='none'])/len(r_syn) * 100)

rr['Argument'].append('synthetic')
rr['Label'].append('fallacious')
rr['Percent'].append(len(r_syn[r_syn.fallacy!='none'])/len(r_syn) * 100)

rr = pd.DataFrame(rr)
plt.figure(figsize=(3.5,3))
ax = sns.barplot(data=rr, x='Argument', y='Percent', hue='Label')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.grid(b=True, which='major', color='w', linewidth=1.0)
ax.grid(b=True, which='minor', color='w', linewidth=0.5)
ax.set(ylim=(0, 100))
plt.tight_layout()
plt.savefig("data/results/fallacy_dist_comparison.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# heatmap 1 x: fallacy y: emotion label: fallacy
hm = defaultdict(list)
sig = {'emotion': defaultdict(list), 'fallacy': defaultdict(list)}

for fallacy, gr in r.groupby('fallacy'):
    hm['fallacy'].append(fallacy)
    ns = []
    sig['fallacy'][fallacy] += [s for s in list(gr['conv_zscore']) if s != 100]
    for emo, g in gr.groupby("emo"):
        sig['emotion'][emo] += [s for s in list(g['conv_zscore']) if s != 100]
        s = np.mean([s for s in list(g['conv_zscore']) if s != 100])
        hm[emo].append(s)
        ns.append(s)

print(len(sig['emotion']))
print(len(sig['fallacy']))

for k, v in sig.items():
    table = defaultdict(list)
    for k1, v1 in v.items():
        var1 = np.var(v1)
        for k2, v2 in v.items():
            var2 = np.var(v2)
            test = ttest_ind(v1, v2, equal_var=True)
            table['cat1'].append(k1)
            table['cat2'].append(k2)
            
            table['difference'].append('larger' if test[0] > 0 else 'smaller')
            table['p-value'].append(test[1])
            table['*'].append("***" if test[1] <= 0.01 else ("**" if test[1] <= 0.05 else ("*" if test[1]<=0.10 else "")))

    table = pd.DataFrame(table)
    table = table[table.cat1!=table.cat2]
    print(f"pairwise t-test for convincingness across {k} categories:")
    print(table)
    table.to_csv(f"data/results/conv_significance_{k}.csv", index=False)
#raise ValueError
    
hm = pd.DataFrame(hm)
hm['avg'] = hm.apply(lambda x: np.mean(x.values[1:]), axis=1)

tmp = pd.DataFrame([['avg']+list(np.mean(hm.values[1:,1:], axis=0))], columns=hm.columns)
hm = pd.concat([hm, tmp], ignore_index=True)
hm.set_index('fallacy', inplace=True)
hm = hm[['enjoyment', 'surprise', 'disgust', 'sadness', 'fear', 'anger', 'none', 'avg']]

# Figure 4 - Average convincingness scores (z-score normalized) for each (perceived) emotion and fallacy category
plt.figure(figsize=(6,4))
ax = sns.heatmap(data=hm, annot=True, cbar=True, fmt='.3f')
ax.set_ylabel(None)
plt.xlabel('Emotion')
plt.ylabel('Fallacy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/results/emo_fallacy_conv_zscore.pdf", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# individuals
# how emotion affects fallacy detection
r = defaultdict(list)
for bi, b in df.groupby('batch'):
    for a in range(3):
        for emo, gr in b.groupby(f"emo_{a}"):
            if batch_annotator_eval[bi][a] >= 0:
                r['Label'].append('Labeled')
                r['Emotion'].append(emo)
                f1 = classification_report(y_pred=gr[f"fallacy_{a}"].tolist(), y_true=gr['fallacy_gold'].tolist(), output_dict=True)['macro avg']['f1-score']
                r['F1'].append(f1)
                
        for emo, gr in b.groupby(f"emotion_gen"):
            if batch_annotator_eval[bi][a] >= 0:
                r['Label'].append('Target')
                r['Emotion'].append(emo if emo!='N/A' else 'none')
                f1 = classification_report(y_pred=gr[f"fallacy_{a}"].tolist(), y_true=gr['fallacy_gold'].tolist(), output_dict=True)['macro avg']['f1-score']
                r['F1'].append(f1)
                
r = pd.DataFrame(r)
label = 'Labeled'
rr = r[r.Label==label]

emotions = list(set(rr['Emotion']))
table = defaultdict(list)
for i in range(len(emotions)):
    for j in range(i+1, len(emotions)):
        g1 = rr[rr.Emotion==emotions[i]]
        g2 = rr[rr.Emotion==emotions[j]]
        data1 = list(g1[g1.Emotion==emotions[i]]['F1'])
        data2 = list(g2[g2.Emotion==emotions[j]]['F1'])
        print(f"size_{emotions[i]}: {len(data1)} vs. size_{emotions[j]}: {len(data2)}: {ttest_ind(data1, data2, equal_var=False if var1/var2 >=4 or var2/var1>=4 else True)}")

        table['emo1'].append(emotions[i])
        table['emo1_mean'].append(np.mean(data1))
        table['emo1_num'].append(len(data1))
        table['emo2'].append(emotions[j])
        table['emo2_mean'].append(np.mean(data2))
        table['emo2_num'].append(len(data2))
        
        test = ttest_ind(data1, data2, equal_var=True)
        table['difference'].append('larger' if test[0] > 0 else 'smaller')
        table['p-value'].append(test[1])
        table['*'].append("***" if test[1] <= 0.01 else ("**" if test[1] <= 0.05 else ("*" if test[1]<=0.10 else "")))

# f1 across perceived emotions + t-test - Table 5+11
table = pd.DataFrame(table)
table.to_csv(f"data/results/f1_significance_{label}.csv", index=False)
print('f1 across perceived emotions + t-test - Table 5+11:')
print(table)

# majority votes
def majority(labels, best_annotator):
    c = Counter(labels).most_common()
    if c[0][1] >= 2:
        return c[0][0]
    else: # if no majority votes, use the annotation from the best annotators
        return labels[best_annotator]

for label in ['emo', 'fallacy']:
    df[f'{label}_final'] = df.apply(lambda x: majority(x[f"{label}"], x[f"{label}_best_annotator"]), axis=1)

df['conv_final'] = df['conv_zscore'].apply(lambda x: np.mean([i for i in x if i!=100]))
df.to_csv("data/annotations/merged_majority.tsv", sep='\t',index=False)

# effects of length difference
df['argument_ori'] = df['id_ori'].apply(lambda x: df[(df.id_ori==x) & (df.id==-1)].iloc[0]['argument'])
df['len_ori'] = df['argument_ori'].apply(lambda x: len(x.split()))
df['len_gen'] = df['argument'].apply(lambda x: len(x.split()))
df['len_diff'] = df.apply(lambda x: x['len_gen']-x['len_ori'], axis=1)
df['len_diff_relative'] = df.apply(lambda x: x['len_diff']/x['len_ori'], axis=1)
binwidth = 5
df['len_diff_group'] = df.apply(lambda x: f"{binwidth*(x['len_diff']//binwidth)} - {binwidth*(x['len_diff']//binwidth)+binwidth}", axis=1)

r = defaultdict(list)
for ld, g in df[(df.id!=-1) & (df.len_diff<=40) & (df.len_diff>0)].groupby("len_diff_group"):
    r['len_ori'].append(np.mean(g['len_ori']))
    r['len_gen'].append(np.mean(g['len_gen']))
    r['len_diff_group'].append(ld)
    r['len_diff'].append(np.mean(g['len_diff']))
    r['num_gen'].append(len(g))
    
    ori = df[(df.id_ori.isin(g['id_ori'].tolist())) & (df.id==-1)]
    r['num_ori'].append(len(ori))
    f1_ori = classification_report(y_pred=ori['fallacy_final'].tolist(), y_true=ori['fallacy_gold'].tolist(),output_dict=True)['macro avg']['f1-score']
    f1_gen = classification_report(y_pred=g['fallacy_final'].tolist(), y_true=g['fallacy_gold'].tolist(),output_dict=True)['macro avg']['f1-score']
    r['f1_diff'].append(f1_gen-f1_ori)
r = pd.DataFrame(r)
print("correlation between length difference and performance difference: ", spearmanr(r['len_diff'], r['f1_diff']))


r_conv, r_f1 = defaultdict(list), defaultdict(list)

for model, gr in df[df.model_gen!='N/A'].groupby("model_gen"):
    r_conv['model'].append(model)
    r_f1['model'].append(model)

    for st, g in gr.groupby('strategy_gen'):
        r_conv[st].append(np.mean(g['conv_final']))
        g = g.sort_values('id')
        r_f1[st].append(classification_report(y_pred=g['fallacy_final'].tolist(), y_true=g['fallacy_gold'].tolist(), output_dict=True)['macro avg']['f1-score'])

r_conv = pd.DataFrame(r_conv)
r_f1 = pd.DataFrame(r_f1)
print('convincingness per emotional framing strategy - Table 4:')
print(r_conv)
conv_ori = np.mean(df[df.model_gen=='N/A']['conv_final'])
print("ori: ", conv_ori)
r_conv.to_csv(f"data/results/conv_model_strategy_ori_{conv_ori}.csv", index=False)

print('F1 per emotional framing strategy - Table 4:')
print(r_f1)
f1_ori = classification_report(y_pred=df[df.model_gen=='N/A']['fallacy_final'].tolist(), y_true=df[df.model_gen=='N/A']['fallacy_gold'].tolist(), output_dict=True)['macro avg']['f1-score']
print("ori: ", f1_ori)
r_f1.to_csv(f"data/results/f1_model_strategy_ori_{f1_ori}.csv", index=False)

