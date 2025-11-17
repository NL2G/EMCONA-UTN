from __init__ import *

df = pd.read_csv("data_cleaned/cleaned_all/merged_masked.csv")
# agreements 
agreements = defaultdict(list)
full = defaultdict(list)
majority = defaultdict(list)

for label in ['conv', 'emo']:
    agg_list = []
    for dataset, group in df.groupby("dataset"):
        n = 0
        m = 0
        total = 0
        for i, batch in group.groupby("batch"):
            batch_labels = list(batch[f"{label}_labels"])
            labels = np.array([[int(ll) for ll in l.split("|")] for l in batch_labels])
            for l in labels:
                if len(set(l)) == 1: # full agreement
                    n += 1
                if Counter(l).most_common()[0][1] >= 3: # majority agreement 
                    m += 1
                total += 1
            majority_labels = list(batch[f'{label}_labels_merged'])
            for j in range(5):
                l1 = labels[:,j]
                for k in range(j+1, 5):
                    l2 = labels[:, k]
                    assert len(l1) == len(l2) == 20
                    kappa = cohen_kappa_score(l1, l2)
                    tmp_labels = labels[:, [j,k]]
                    kripppen = kd.alpha(tmp_labels.T, level_of_measurement='nominal')
                    agreements['label'].append(label)
                    agreements['dataset'].append(dataset)
                    agreements['batch'].append(i)
                    agreements['annotators'].append(f"a{j}-{k}")
                    agreements['kappa'].append(kappa)
                    agreements['krippen'].append(kripppen)
        full['dataset'].append(dataset)
        full['label'].append(label)
        assert total == 200, total
        full['full_n'].append(n/total)
        full['majority_n'].append(m/total)

full = pd.DataFrame(full)
print(full)
full.to_csv("data_cleaned/cleaned_all/majority_agreements.csv", index=False)
agreements = pd.DataFrame(agreements) 
print(agreements)
print(agreements.groupby(['dataset', 'batch', 'label']).max())
agreements = agreements.groupby(['dataset', 'batch', 'label']).max() # find the most agreeing annotator pairs for each batch
agreements = agreements.reset_index()
print(agreements)
agreements = agreements.groupby(['label', 'dataset']).mean().reset_index() # average over batches per dataset
print(agreements)
agreements.to_csv("data_cleaned/cleaned_all/most_agreeing.csv", index=False)
