from __init__ import *

def categorize(labels):
    assert len(labels) == 2, labels
    if labels[0] == labels[1]: 
        return "c"
    elif labels[0] == 1 and labels[1] in [0, 2]:
        return "p"
    elif labels[0] == 0:
        if labels[1] == 1:
            return "n"
        else:
            return "p"
    elif labels[0] == 2 and labels[1] in [0, 1]:
        return "n"
    else:
        raise ValueError
    
def realign(ids, label):
    if label == 0:
        return label
    if ids in ['0,1', '0,2', '3,1', '3,2']:
        return label
    else:
        return 3-label
    
mapids = {
    "1,0": "0,1",
    "2,0": "0,2",
    "1,3": "3,1",
    "2,3": "3,2"
}


if __name__ == "__main__":
    # bws
    df = pd.read_csv("data_cleaned/cleaned_all/merged_masked.csv")
    df['dataset'] = df['id'].apply(lambda x: x.split("-")[0])
    df['conv'] = df['conv_labels'].apply(lambda x: [int(j) for j in x.split("|")])
    df['emo'] = df['emo_labels'].apply(lambda x: [int(j) for j in x.split("|")])

    bws_df = defaultdict(list)
    idl = [["1,0", "0,1"], ["1,3", "3,1"], ["2,0", "0,2"], ["2,3", "3,2"]]

    for dataset, group in df.groupby("dataset"):
        bws = {"conv": defaultdict(list), "emo": defaultdict(list)}
        for id, tmp in group.groupby(by='id'):
            assert len(tmp) == 4, group
            for indices in idl:
                row = tmp[tmp.ids.isin(indices)].iloc[0]
                ids = row['ids'].split(",")
                for j, label in enumerate(['conv', 'emo']):
                    if row[f'{label}_labels_merged'] == 1:
                        bws[label][ids[0]].append(1)
                        bws[label][ids[-1]].append(-1)
                    elif row[f'{label}_labels_merged'] == 2:
                                bws[label][ids[0]].append(-1)
                                bws[label][ids[-1]].append(1)
                    elif row[f'{label}_labels_merged'] == 0:
                        bws[label][ids[0]].append(0)
                        bws[label][ids[-1]].append(0)
                    else:
                        raise ValueError("???")
        bwsf = {"conv": defaultdict(float), "emo": defaultdict(float)}
        for label, b in bws.items():
            for system, count in b.items():
                best = len([c for c in count if c == 1])
                worst = len([c for c in count if c == -1])
                l = len(count)
                bwsf[label][system] = (best - worst) / l
        
        for system in bwsf[label].keys():
            bws_df['dataset'].append(dataset)
            bws_df['system'].append(system)
            for label in ['emo', 'conv']:
                bws_df[label].append(bwsf[label][system])
                
    bws_df = pd.DataFrame(bws_df).sort_values(['dataset', 'system'])
    print(bws_df)
    bws_df.to_csv("data_cleaned/cleaned_all/bws.csv", index=False)


    df['ids_aligned'] = df['ids'].apply(lambda x: mapids[x] if x not in['0,1', '0,2', '3,1', '3,2'] else x)
    final = []
    for dataset, tmp in df.groupby('dataset'):
        for _, group in tmp.groupby("id"):
            assert len(group) == 4, group
            annotators = group.iloc[0]['annotators'].split("|")
            for i in range(len(group.iloc[0]['conv'])): # individual
                group['conv_tmp'] = group.apply(lambda x: realign(x['ids'], x['conv'][i]), axis=1)
                group['emo_tmp'] = group.apply(lambda x: realign(x['ids'], x['emo'][i]), axis=1)
                anchor = group[group.ids_aligned.isin(["0,1","1,0"])].iloc[0:1]
                others = group.drop(index=anchor.index).sort_values("ids")
                assert len(anchor) == 1
                anchor = anchor.iloc[0]

                for _, other in others.iterrows():
                    category = categorize([anchor['conv_tmp'], other['conv_tmp']]) # use aligned labels
                    final.append(pd.DataFrame({"annotator": annotators[i], "id": [other["id"]], "dataset": dataset, "ids": other['ids_aligned'], "category": category, "emotion_other": other['emo_tmp'], "emotion_anchor": anchor['emo_tmp']}))

    final = pd.concat(final, ignore_index=True)
    assert len(final) == 3750 # len: 3750 = 5 annotators x 50 test instances x 3 counterpart pairs x 5 datasets
    #print(final)
    final.to_csv("data_cleaned/cleaned_all/human_cat.csv", index=False)

    rr = defaultdict(list) # rates table
    for (a, dataset), group in final.groupby(["annotator", 'dataset']):
        for c in ['c', 'p', 'n']:
            rates = []
            for id, tmp in group.groupby('id'):
                assert len(tmp) == 3
                rate = sum([ca == c for ca in tmp['category']]) / len(tmp)
                rates.append(rate*100)
                rr['dataset'].append(dataset)
                rr['annotator'].append(a)
                rr['category'].append(c)
                rr['id'].append(id)
                rr['rate'].append(rate * 100)
            rr['dataset'].append(dataset)
            rr['annotator'].append(a)
            rr['category'].append(c)
            rr['id'].append(f'average')
            rr['rate'].append(np.mean(rates))
    rr = pd.DataFrame(rr)
    #print(rr)
    rr.to_csv("data_cleaned/cleaned_all/rates.csv", index=False)

    d2d = {
    'deuparl': "DeuParl$_{de}$",
    'hansard': "Hansard$_{en}$",
    'hansard_bill': 'Bill$_{en}$',
    'dagstuhl': "Dagstuhl$_{en}$",
    'defabel': "Defabel$_{de}$"
    }

    d2l = {
        'deuparl': "de",
        'hansard': "en",
        'hansard_bill': 'en',
        'dagstuhl': "en",
        'defabel': "de"
    }

    c2c = {
        'p': 'Positivity',
        'c': 'Consistency',
        'n': 'Negativity'
    }

    rr['language'] = rr['dataset'].apply(lambda x: d2l[x])
    rr['dataset'] = rr['dataset'].apply(lambda x: d2d[x])
    rr['category'] = rr['category'].apply(lambda x: c2c[x])
    rr = rr.sort_values(['language', 'category'], ascending=False)

    
    plt.figure(figsize=(5,4))
    sns.barplot(data=rr[rr.id=='average'], y='rate', x='dataset', hue='category', errorbar=('ci', 95))
    plt.xticks(rotation=30, fontsize=10)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Rate (%)", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/rate_r7.pdf", dpi=300, bbox_inches='tight')
    plt.show()     

    
     

    
