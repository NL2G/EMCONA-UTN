from __init__ import *

paths = sorted(glob("data_cleaned/downloaded_round7/*/forms/*batch_*（回复）.xlsx"))

final = []
fails = defaultdict(list)

name2id = {
    'Yanran': "a1",
    'Lisanne': "a2",
    "Fabian": "a3",
    "Tianqi Luan": "a4",
    "Tianqi": "a4"
}

for path in paths:
    print(path)
    _, _, dataset, _, batch = path.split("/")
    batch = batch.split("（")[0]

    df = pd.read_excel(path)
    df = df[~df[df.columns[-1]].isna()]
    if dataset == 'dagstuhl' and batch in ['batch_1', 'batch_3']: # extra annotations received --> removed
        df = df.iloc[:-1]

    df['Please enter your Prolific ID'] = df['Please enter your Prolific ID'].apply(lambda x: name2id[x] if x in name2id.keys() else x)
    tmp = df.iloc[:, 26:]
    for col in df.columns[26: ]:
        df[col] = df[col].apply(lambda x: int(x.split()[0])) # label 2 int number

    # attention check columns: 33, 41, 58
    f = [] # id failed
    for _, row in df.iterrows():
        if row['Please enter your Prolific ID'] in name2id.values():
            continue
        failed = 0
        tmp = row.values[33]
        if tmp != 1:
            failed += 1
        tmp = row.values[41]
        if tmp != 2:
            failed += 1
        tmp = row.values[58]
        if tmp != 0:
            failed += 1

        fails['id'].append(row['Please enter your Prolific ID'])
        fails['dataset'].append(dataset)
        fails['batch'].append(batch)
        if failed >= 2:
            fails['failed'].append(True)
            fails['score'].append(f"{failed}/3")
            f.append(row['Please enter your Prolific ID'])
        else:
            fails['failed'].append(False)
            fails['score'].append(f"{failed}/3")

    df = df[~df['Please enter your Prolific ID'].isin(f)] # remove annotations with failed attention checks
    len1 = len(df)  
    df = df.drop_duplicates('Please enter your Prolific ID')
    len2 = len(df)
    if len1 != len2:
        print("repeated entry:")
        print(path)

    path1 = path.replace("（回复）.xlsx", "_form.tsv")
    df1 = pd.read_csv(path1, sep='\t')

    df1['labels'] = df.iloc[:, 26:].values.T.tolist()
    df1['annotators'] = [list(df['Please enter your Prolific ID'])] * len(df1)
    df1 = df1[df1.id!='check'] # attention check items

    path2 = "/".join(path.split("/")[:3])+f"/{batch}.tsv"
    df = pd.read_csv(path2, sep='\t')
    df['conv_labels'] = list(df1.iloc[:20]['labels'])
    df['emo_labels'] = list(df1.iloc[20:]['labels'])
    df['annotators'] = list(df1.iloc[:20]['annotators'])
    df['batch'] = [batch] * len(df)
    df.drop(columns=['instance', 'arg1', 'arg2'], inplace=True)

    #print(df)
    final.append(df)
    #raise ValueError

fails = pd.DataFrame(fails).sort_values("failed")
print("Attention check failed: ", len(fails[fails.failed==True])/len(fails))

df = pd.concat(final, ignore_index=True).sort_values('id')

# add student annotations from spreadsheets
# hansard bill
tmp = pd.read_csv("data/hansard_annotation/emcona_3/merged_49_tuples_new.tsv", sep='\t')
tmp.rename(columns={"tuple_index": "index", "Argument 1": "arg1", "Argument 2": "arg2"}, inplace=True)
tmp['ids'] = tmp['ids'].apply(lambda x: x.replace("[","").replace("]","").replace(", ", ","))
tmp['index'] = tmp['index'].apply(lambda x: f"hansard_bill-{x}")

for i, row in tmp.iterrows():
    emo = row['emotion'] if row['emotion'] != 3 else 0 # previously we had 0 and 3 labels indicating both non and equal
    conv = row['convincingness'] if row['convincingness'] != 3 else 0
    a = row['annotator']
    ids = row['ids']
    id = row['index']

    index = df[(df.id==id) & (df.ids==ids)].index
    assert len(index) == 1, index
    index = index[0]
    assert len(a) == 2, a # two chars

    df.loc[index, 'emo_labels'] = df.loc[index, 'emo_labels'] + [emo]
    df.loc[index, 'conv_labels'] = df.loc[index, 'conv_labels'] + [conv]
    df.loc[index, 'annotators'] = df.loc[index, 'annotators'] + [a]

    assert len(df.loc[index]['emo_labels']) == len(df.loc[index]['conv_labels']) == len(df.loc[index]['annotators'])

# deuparl
paths = sorted(glob("data/deuparl_annotation/emcona/*_2.tsv"))
final = []
for path in paths:
    tmp = pd.read_csv(path, sep='\t')
    a = path.split("/")[-1][:2]
    if "a3" in path and "round8" in path:
        tmp2 = pd.read_csv(path.replace("round8_2", "round7"), sep='\t')
        tmp['emotion'] = tmp['index'].apply(lambda x: tmp2[tmp2['index']==x].iloc[0]['emotion'])
    else:
        tmp2 = pd.read_csv(path.replace("_2", "_1"), sep='\t')
        tmp.rename(columns={"emotion (Which argument evokes stronger emotions in you?)": "emotion"}, inplace=True)
        tmp['convincingness'] = list(tmp2['convincingness'])

    if "round9" in path:
        tmp['ids'] = ['0,2'] * len(tmp)
    elif "round10" in path: 
        tmp['ids'] = ['1,3'] * len(tmp)
    elif "round11" in path: 
        tmp['ids'] = ['2,3'] * len(tmp)

    tmp['annotator'] = [a] * len(tmp)
    tmp['id'] = tmp['index'].apply(lambda x: f"deuparl-{x}")
    tmp = tmp[['id', 'ids', 'emotion', 'convincingness', 'annotator']]
    final.append(tmp)

tmp = pd.concat(final, ignore_index=True)
tmp['convincingness'] = tmp['convincingness'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else int(x))
tmp['emotion'] = tmp['emotion'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else int(x))


for i, row in tmp.iterrows():
    emo = row['emotion']
    conv = row['convincingness']
    a = row['annotator']
    ids = row['ids']
    id = row['id']

    index = df[(df.id==id) & (df.ids==ids)].index
    if len(index) == 0:
        new_row = pd.DataFrame({'id': id, 'lang': 'de', 'ids': ids, 'conv_labels': [[conv]], 'emo_labels': [[emo]], 'annotators': [[a]], 'batch': None})
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        index = index[0]

        assert len(a) == 2, a 
        df.loc[index, 'emo_labels'] = df.loc[index, 'emo_labels'] + [emo]
        df.loc[index, 'conv_labels'] = df.loc[index, 'conv_labels'] + [conv]
        df.loc[index, 'annotators'] = df.loc[index, 'annotators'] + [a]

        assert len(df.loc[index]['emo_labels']) == len(df.loc[index]['conv_labels']) == len(df.loc[index]['annotators'])

df['dataset'] = df['id'].apply(lambda x: x.split("-")[0])
df = df.sort_values(["id", "ids"])

# check progress

for dataset, group in df.groupby('dataset'):
    num_annotators = [len(x) for x in list(group['emo_labels'])]
    print(np.mean(num_annotators))
    print(f"One instance received {np.mean(num_annotators)} annotations in {dataset} on average.")
    for batch, g in group.groupby('batch'):
        num_annotators = [len(x) for x in list(g['emo_labels'])]
        if np.mean(num_annotators) != 5:
            print()
            print(dataset)
            print(batch)
            print(np.mean(num_annotators))


def majority_vote(labels, majority=True):
    if majority:
        c = Counter(labels).most_common()
        if c[0][1] >= len(labels)//2 + 1:
            return c[0][0]
        else:
            return 0
    else:
        c = Counter(labels)
        score = c[2]-c[1]
        return 1 if score < 0 else (0 if score==0 else 2)
    
for col in ['conv_labels', 'emo_labels',  'annotators']:
    if "label" in col:
        df[f"{col}_merged"] = df[col].apply(lambda x: majority_vote(x))
        df[f"{col}_merged_score"] = df[col].apply(lambda x: majority_vote(x, False))
    df[col] = df[col].apply(lambda x: "|".join([str(j) for j in x]))

df.to_csv("data_cleaned/cleaned_all/merged.csv", index=False)

# mask Prolific ID
id2mask = {}
alist = []
for a in list(df['annotators']):
    a = a.split("|")
    newa = []
    for aa in a:
        if aa not in id2mask.keys():
            id2mask[aa] = len(id2mask)
        newa.append(str(id2mask[aa]))
    newa = "|".join(newa)
    alist.append(newa)

df['annotators'] = alist
df.to_csv("data_cleaned/cleaned_all/merged_masked.csv", index=False)

