from __init__ import *

mapids = {
    "1,0": "0,1",
    "2,0": "0,2",
    "1,3": "3,1",
    "2,3": "3,2"
}

def realign(ids, label):
    if label == 0:
        return label
    if ids in ['0,1', '0,2', '3,1', '3,2']:
        return label
    else:
        return 3-label

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
        print('invalid labels: ')
        print(labels)
        return None
        #raise ValueError

def decode(df):
    df['emo'] = [[]] * len(df)
    df['conv'] = [[]] * len(df)
    emos, convs = [], []
    #n = int((len(df.columns)-4)/2)
    #assert n in [3,5]
    for index, row in df.iterrows():
        emo, conv = [], []
        
        for i in range(5):
            """
            match = re.search(r"\d", str(row[f'emo_{i+1}']))
            if match:
                label = int(match.group())
                label = realign(row['ids'], label)
                if label in [0, 1, 2]:
                    emo.append(int(label))
                else:
                    emo.append(None)
                #df.loc[index, 'emo'].append(label)
            else:
                emo.append(None)
                #df.loc[index, 'emo'].append(None)
            """
            match = re.search(r"\d", str(row[f'conv_{i+1}']))
            if match:
                label = int(match.group())
                label = realign(row['ids'], label)
                
                if label in [0, 1, 2]:
                    conv.append(int(label))
                else:
                    conv.append(None)
            else:
                conv.append(None)

        emos.append(emo)
        convs.append(conv)
        
    #df['emo'] = emos
    #df['conv'] = convs
    #df['emo_merged'] = majority(emos)
    df['conv_merged'] = [majority(l)[0] for l in convs]
    #df = df[['id', 'ids', 'emo', 'conv']]
    #df = df[['id', 'ids', 'conv', 'conv_merged']]
    df = df[['id', 'ids', 'conv_merged']]
    df['ids_aligned'] = df['ids'].apply(lambda x: mapids[x] if x not in['0,1', '0,2', '3,1', '3,2'] else x)
    # label is not aligned according to ids
    return df

def majority(l):
    c = Counter(l).most_common()
    #print(l)
    if c[0][1] >= len(l)//2 + 1:
        return c[0][0], c[0][1]/len(l)
    else:
        #print(l)
        #return None, 0 where is None used?
        return 0, 0

def majority_c(c):
    c = Counter(c).most_common()
    if c[0][1] >= len(l)//2 + 1:
        return c[0][0]
    else:
        return 'c'
    
def calculate(df):
    final = []
    for _, group in df.groupby("id"):
        anchor = group[group.ids_aligned=='0,1'].iloc[0:1]
        others = group.drop(index=anchor.index).sort_values("ids")
        assert len(anchor) == 1
        anchor = anchor.iloc[0]
        #n = len(anchor['conv'])
        #assert n == 5
        for _, other in others.iterrows():
            # final label based on majority
            #print(anchor)
            #print(other)
            #print(path)
            category = categorize([anchor['conv_merged'], other['conv_merged']])
            final.append(pd.DataFrame({"id": [other["id"]], "ids": other['ids_aligned'], "category": category, 'round': '-1'}))
            """
            for i in range(n):
                if anchor['conv'][i] is None or other['conv'][i] is None:
                    continue
                category = categorize([anchor['conv'][i], other['conv'][i]])    
                final.append(pd.DataFrame({"id": [other["id"]], "ids": other['ids_aligned'], "category": category, 'round': i+1}))
            """
    final = pd.concat(final, ignore_index=True)
    
    return final

if __name__ == "__main__":
    d2l = {
    'deuparl': "de",
    'hansard': "en",
    'hansard_bill': 'en',
    'dagstuhl': "en",
    'defabel': "de"
    }
    
    b2s = {
        'large': ['Qwen2.5-72B-Instruct', 'Mixtral-8x7B-Instruct-v0.1', 'Llama-3.3-70B-Instruct', 'gpt-4o-2024-08-06'],
        'middle': ['Llama-3.2-3B-Instruct', 'Qwen2.5-7B-Instruct', 'Mistral-7B-Instruct-v0.3', 'gpt-3.5-turbo'],
        'small': ['Qwen2.5-0.5B-Instruct', 'Llama-3.2-1B-Instruct', 'gpt-4o-mini']
    }
    
    # llms
    llm = []
    row_data = []
    for path in glob("data_cleaned/llms/llms/*.tsv"):
        final = defaultdict(list)
        tmp = pd.read_csv(path, sep='\t')
        tmp = decode(tmp)
        tmp_cat = calculate(tmp)
        assert len(tmp_cat) == 750, tmp_cat # 50 instances x 3 counterpartpairs x 5 datasets = 750
 
        prompt = path.split("/")[-1][-5]
        checkpoint = path.split("/")[-1][:-16] #if 'Mixtral' not in path else 'Mistral'
        
        #size = [k for k, v in b2s.items() if checkpoint in v]
        checkpoint = checkpoint #if 'Mixtral' not in path else 'Mistral'
        #family = checkpoint.split("-")[0]
        #if len(size) != 1:
        #    print(size)
        #    size = None
        #else:
        #    size = size[0]
        #n = 5#3 if 'gpt' in checkpoint else 5

        #tmp['dataset'] = tmp['id'].apply(lambda x: x.split("-")[0])
        #tmp['model'] = checkpoint
        #tmp['prompt'] = prompt

        tmp_cat['dataset'] = tmp_cat['id'].apply(lambda x: x.split("-")[0])
        tmp_cat['model'] = checkpoint
        tmp_cat['prompt'] = int(prompt)
        
        row_data.append(tmp_cat)
        
    row_data = pd.concat(row_data, ignore_index=True)
    assert len(row_data) == 24750, row_data # 11 models x 750 results x 3 prompts 
    row_data.to_csv("data_cleaned/cleaned_all/row_data.csv", index=False)

    
    #rr = pd.read_csv("data_cleaned/llms/row_data.csv")
    llm_rate = defaultdict(list)
    for id, group in row_data.groupby('id'):
        for (model,prompt,round), t in group.groupby(['model', 'prompt', 'round']):
            c = Counter(list(t['category']))
            #print(c)
            for category in ['c', 'n', 'p']:
                llm_rate['model'].append(model)
                llm_rate['prompt'].append(prompt)
                llm_rate['round'].append(round)
                llm_rate['id'].append(id)
                llm_rate['category'].append(category)
                llm_rate['rate'].append(c[category]/3 if category in c.keys() else 0)
        
    llm_rate = pd.DataFrame(llm_rate)
    #print(llm_rate)
    c2c = {
        'c': 'Consistency',
        'p': 'Positivity',
        'n': 'Negativity'
    }
    llm_rate['category'] = llm_rate['category'].apply(lambda x: c2c[x])
    llm_rate = llm_rate.groupby(['model', 'prompt', 'round','category']).mean().reset_index()
    #llm_rate['model'] = llm_rate['model'].apply(lambda x: 'Mixtral-8x7B-Instruct-v0.1' if model == 'Mistral' else x)
    llm_rate['rate'] = llm_rate['rate'] * 100

    plt.figure(figsize=(15,4))
    #sns.barplot(data=llm_rate[llm_rate.prompt==1], x='model', y='rate', hue='category')
    sns.barplot(data=llm_rate, x='model', y='rate', hue='category', hue_order=['Positivity', 'Negativity', 'Consistency'], errorbar=None)
    plt.xticks(rotation=10, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="upper left", fontsize=11, bbox_to_anchor=(1, 1))
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Rate (%)', fontsize=12)

    plt.tight_layout()
    #plt.savefig("plots/llms/dis_all_prompts.pdf", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    
    for prompt in range(1,4):
        plt.figure(figsize=(15,5))
        #sns.barplot(data=llm_rate[llm_rate.prompt==1], x='model', y='rate', hue='category')
        
        sns.barplot(data=llm_rate[llm_rate.prompt==prompt], x='model', y='rate', hue='category', hue_order=['Positivity', 'Negativity', 'Consistency'], errorbar=None)
        plt.xticks(rotation=10, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Rate (%)', fontsize=12)
        plt.title(f'Prompt {prompt}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"plots/llms/dis_prompt{prompt}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    #raise ValueError
    
    

    final = defaultdict(list)
    #llm = pd.read_csv("data_cleaned/llms/row_data.csv")
    llm = row_data

    human = pd.read_csv("data_cleaned/cleaned_all/merged_masked.csv")
    human['conv_m'] = human.apply(lambda x: realign(x['ids'], x['conv_labels_merged']), axis=1)
    human = human.sort_values(['id', 'ids'])

    human_cat = defaultdict(list)

    for id, tmp in human.groupby('id'):
        anchor = tmp[tmp.ids.isin(["0,1", "1,0"])]
        others = tmp.drop(index=anchor.index)
        anchor = anchor.iloc[0]
        for _, row in others.iterrows():
            category = categorize([anchor['conv_m'], row['conv_m']])
            human_cat['dataset'].append(row['dataset'])
            human_cat['id'].append(id)
            human_cat['ids'].append(row['ids'])
            human_cat['category'].append(category)
    
    human_cat = pd.DataFrame(human_cat)
    human_cat['ids'] = human_cat['ids'].apply(lambda x: mapids[x] if x not in['0,1', '0,2', '3,1', '3,2'] else x)
    human_cat['idids'] = human_cat.apply(lambda x: f"{x['id']}/{x['ids']}", axis=1)
    llm['idids'] = llm.apply(lambda x: f"{x['id']}/{x['ids']}", axis=1)
    print(len(llm))
    llm = llm.dropna(subset='category')
    print("remove non: ", len(llm))
   
    s = defaultdict(list)
    for (dataset, model, prompt), group in llm.groupby(["dataset", "model", "prompt"]):
        h = human_cat[human_cat.idids.isin(list(group['idids']))]
        l = []
        group = group.sort_values(['id', 'ids'])
        h = h.sort_values(['id', 'ids'])
        if len(group) != len(h):
            print('rerun')
            print(model)
            print(prompt)
            print(dataset)
            print(len(group))
            print(len(h))
            s['model'].append(model)
            s['prompt'].append(prompt)
            s['dataset'].append(dataset)
            s['len'].append(len(group))
            raise ValueError
        assert len(h) == len(group)
        #print(h)
        #print(group)
        
        kappa = cohen_kappa_score(list(h['category']), list(group['category']))
        final['dataset'].append(dataset)
        final['model'].append(model)
        final['prompt'].append(prompt)
        final['kappa'].append(kappa)
        f1 = classification_report(y_true=list(h['category']), y_pred=list(group['category']), output_dict=True)['macro avg']['f1-score']
        final['f1'].append(f1)

    final = pd.DataFrame(final)
    s = pd.DataFrame(s)
    s.to_csv("data_cleaned/llms/rerun.csv", index=False)
    final.to_csv('data_cleaned/cleaned_all/llm_kappa_cate.csv', index=False)

    #raise ValueError
    
    final = defaultdict(list)
    for path in glob("data_cleaned/llms/llms/*.tsv"):
        tmp = pd.read_csv(path, sep='\t')
        tmp = decode(tmp)
        tmp['dataset'] = tmp['id'].apply(lambda x: x.split("-")[0])
        prompt = path.split("/")[-1][-5]
        checkpoint = path.split("/")[-1][:-16] 
        tmp = tmp.sort_values(['id', 'ids'])
        
        for dataset, h in human.groupby("dataset"):
            l = tmp[tmp.dataset==dataset]
            for label in ['conv']:   
                idx = [i for i, ll in enumerate(list(l[f"{label}_merged"])) if np.isnan(ll)]
                llabels = [ll for i, ll in enumerate(list(l[f"{label}_merged"])) if i not in idx]
                hlabels = [ll for i, ll in enumerate(list(h[f"{label}_m"])) if i not in idx]

                kappa = cohen_kappa_score(hlabels, llabels)
                #print(classification_report(y_true=hlabels, y_pred=llabels))
                f1 = classification_report(y_true=hlabels, y_pred=llabels, output_dict=True)['macro avg']['f1-score']
                final['dataset'].append(dataset)
                final['model'].append(checkpoint)
                final['prompt'].append(prompt)
                final['label'].append(label)
                final['kappa'].append(kappa)
                final['f1'].append(f1)

        #print(final)
        
    final = pd.DataFrame(final)
    #print(final)
    final.to_csv("data_cleaned/cleaned_all/llm_kappa_ind.csv", index=False)
    #raise ValueError
    

    llm1 = pd.read_csv("data_cleaned/cleaned_all/llm_kappa_ind.csv")
    llm1['language'] = llm1['dataset'].apply(lambda x: 'de' if x in ['deuparl', 'defabel'] else 'en')
    llm1 = llm1[llm1.label=='conv']
    llm1 = llm1.groupby(['model', 'prompt', 'language']).mean().reset_index()

    llm2 = pd.read_csv("data_cleaned/cleaned_all/llm_kappa_cate.csv")
    llm2['language'] = llm2['dataset'].apply(lambda x: 'de' if x in ['deuparl', 'defabel'] else 'en')
    llm2 = llm2.groupby(['model', 'prompt', 'language']).mean().reset_index()
    

    final = defaultdict(list)


    # all datasets
    for lang, group in llm1.groupby('language'):
        for model, tmp in group.groupby("model"):
            final['language'].append(lang)
            final['model'].append(model)
            assert len(tmp) == 3, tmp
            avg = np.mean(tmp['f1'])
            std = np.std(tmp['f1'])

            s = f"{avg:.3f}±{std:.3f}"
            final['Static'].append(f"{np.max(tmp['f1']):.3f}")
            final['Static_ranking'].append(np.max(tmp['f1']))

            tmp2 = llm2[(llm2.language==lang) & (llm2.model==model)]
            assert len(tmp2) == 3, tmp2
            avg = np.mean(tmp2['f1'])
            std = np.std(tmp2['f1'])

            s = f"{avg:.3f}±{std:.3f}"
            final['Dynamic'].append(f"{np.max(tmp2['f1']):.3f}")
            final['Dynamic_ranking'].append(np.max(tmp2['f1']))

    final = pd.DataFrame(final)

    for lang, group in final.groupby('language'):
        group = group.sort_values("Dynamic_ranking", ascending=False)
        group['Dynamic_ranking'] = [i for i in range(1, len(group)+1)]
        group = group.sort_values("Static_ranking", ascending=False)
        group['Static_ranking'] = [i for i in range(1, len(group)+1)]
        print(lang)
        print(group)
        group.to_csv(f'data_cleaned/cleaned_all/llm_rankings_{lang}.csv', index=False)



