from utils import *
from templates import *

ori = pd.read_csv("data_cleaned/merged_1000.tsv", sep='\t')#.iloc[:1]

pbar = tqdm(total=2*3*3*1000) # 2 rounds

for p in ['chat_1', 'chat_2', 'chat_3']:
    for model in ["gpt-4o-2024-08-06", "gpt-3.5-turbo",  "gpt-4o-mini"]:
        df = ori
        for i in range(2):
            emo_labels, conv_labels = [], []
            for _, row in df.iterrows():
                # no emo needed
                label = conv_comparison(prompt=prompt[p][1].format(text=row['instance']), model=model)
                conv_labels.append(label)
                pbar.update(1)
            df[f'conv_{i+1}'] = conv_labels
        cols = [c for c in df.columns if "emo" in c or "conv" in c]
        df = df[['id', 'ids'] + cols]
        df.to_csv(f"data_cleaned/llms/{model.split('/')[-1]}_1000_{p}.tsv", index=False, sep='\t')

