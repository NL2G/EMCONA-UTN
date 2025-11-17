
from __init__ import *
from utils import *

# defabel new 
df = pd.read_csv("data/defabel_annotation/sample_50.tsv", sep='\t')
print(df)

c1, c2, c3 = [], [], []
rephrase = {}
ids = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    if row['diff2'] > 0:
        r0 = row['arg1']
        r1 = row['arg2']
        ids.append("0,1")
    else:
        r0 = row['arg2']
        r1 = row['arg1']
        ids.append("1,0")

    # add
    if r1 not in rephrase.keys():
        r2 = add_emotion(text=r1, lang='de', threshold=85, patience=3)
        rephrase[r1] = {'id': 2, 'text': r2[0], 'label': r2[1], 'likelihood': r2[2]}
        r2 = r2[0]
    else:
        r2 = rephrase[r1]['text']
        
    if r0 not in rephrase.keys():
        r3 = remove_emotion2(text=r0, lang='de', threshold=85, patience=3)
        rephrase[r0] = {'id': 3, 'text': r3[0], 'label': r3[1], 'likelihood': r3[2]}
        r3 = r3[0]

    else:
        r3 = rephrase[r0]['text']
    # pair 
    # r0 vs. r2
    c1.append({'index': row['index'], 'topic': row['topic'], 
               'arg1': r0, 'arg2': r2, 'ids': "0,2"})

    # r1 vs. r3
    c2.append({'index': row['index'], 'topic': row['topic'], 
               'arg1': r1, 'arg2': r3, 'ids': "1,3"})

    # r2 vs. r3
    c3.append({'index': row['index'], 'topic': row['topic'], 
               'arg1': r2, 'arg2': r3, 'ids': "2,3"})

with open("data/defabel_annotation/rephrase.json", 'w') as f:
    json.dump(rephrase, f, indent=2)

df['ids'] = ids
df = df[['index', 'ids', 'topic', 'arg1', 'arg2']]   
c1 = pd.DataFrame(c1).sample(frac=1, random_state=32)
c2 = pd.DataFrame(c2).sample(frac=1, random_state=41)
c3 = pd.DataFrame(c3).sample(frac=1, random_state=2)
df = pd.concat([df, c1, c2, c3], ignore_index=True).sort_values("index")

df.to_csv("data/defabel_annotation/200.tsv", sep='\t')
