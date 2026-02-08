from parallel_api_call import *

def custom_parse_response(s):
        r = {}
        if "===" in s:
            for argument in s.split("==="):
                argument = argument.strip()
                emotion = argument.split(":")[0][9:]
                argument = ":".join(argument.split(":")[1:])
                r[emotion] = argument
        else:
            for argument in s.split("\n\n"):
                argument = argument.strip()
                emotion = argument.split(":")[0][9:]
                argument = ":".join(argument.split(":")[1:])
                r[emotion] = argument
        if len(r)!=6:
            for k in ['anger', 'surprise', 'disgust', 'enjoyment', 'fear', 'sadness']:
                if k not in r.keys():
                    r[k] = None
            for k in list(r.keys()):
                if k not in ['anger', 'surprise', 'disgust', 'enjoyment', 'fear', 'sadness']:
                    del r[k]
        assert len(r)==6, r.keys()
        return r
        #pass

def custom_apply_template(x):
    methods = """There are many methods to add emotional appeal—also known as pathos—to an argument, but here are some of the most common and effective ones:
1. Storytelling – Sharing personal stories or anecdotes that evoke emotion.
2. Vivid Language – Using descriptive, sensory, or emotionally charged words.
3. Loaded Words – Choosing words with strong positive or negative connotations.
4. Imagery – Painting mental pictures to stir emotions."""
    method_detail = {
    'storytelling': "keep the original argument but **only** add one sentence",
    'vivid language': "keep the original argument but **only** change its word choices",
    'loaded words': "keep the original argument but **only** change its word choices",
    'imagery': "keep the original argument but **only** add one sentence"
}
    prompt = """I will give an argument. Your task is to add emotional appeals to the given argument by using **{method}**. You should {explain} to change its emotional tone. You generate 6 arguments, each with one kind of emotion: *anger*, *surprise*, *disgust*, *enjoyment*, *fear*, and *sadness*. Note: Please put the modified parts in bold and do **NOT** use emotionally charged words in the added sentence when using storytelling and imagery.

Answer in the following way:
argument_anger: [the generated argument that appeals to anger]
===
argument_suprise: [the generated argument that appeals to suprise]
===
...

Argument: {argument}""".format(argument=x['inputs_0'], method=x['inputs_1'], explain=method_detail[x['inputs_1']])
    prompts = [
        {'role': 'system',
        'content': methods
        },
        {'role': 'user',
        'content': prompt}
    ]
  
    return prompts

if __name__ == "__main__":
    

    parser = ArgumentParser()
    parser.add_argument("--file", '-f', type=str, help="The tsv file to be evaluated.", default=None)
    parser.add_argument("--model", '-m', type=str, help="The LLM used for generation", default='openai/gpt-4o-mini')
    parser.add_argument("--api_base", type=str, help="The API link.", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api_key", type=str, help='The local txt file storing the API key.', default="openrouter_key.txt")
    parser.add_argument("--cache_dir", type=str, help='The cache directory.', default="./cache/")
    parser.add_argument("--concurrency", type=int, help='The parrallel processing number.', default=50)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--out_dir", type=str, help='The cache directory.', default="./outputs2b_611")
    parser.add_argument("--method", type=str, help='The strategy used to evoke emotions.', default=None)

    args = parser.parse_args()
    
    if args.file is not None:
        df = pd.read_csv(args.file, sep='\t')
    else:
        df = pd.read_csv("edu_all.csv")
        df.dropna(subset="source_article", inplace=True)
        df = df[df.updated_label.isin(["ad hominem", "faulty generalization", "false causality", "circular reasoning", "ad populum"])]
        df = df[~df.source_article.str.lower().str.contains('fallacy') & ~df.source_article.str.lower().str.contains('fallacies') 
                & ~df.source_article.str.lower().str.contains('this is an example of') & ~df.source_article.str.lower().str.contains(' x ')]
        df.drop_duplicates('source_article', inplace=True)
    
    if args.test:
        df = df.iloc[:5]

    paras = {k:v for k,v in vars(args).items() if k not in ['file', 'test', 'out_dir', 'method']}


    api = APICall(**paras, parse_response=custom_parse_response, apply_template=custom_apply_template)
    
    new_df = defaultdict(list)
  
    df = df[df.is_argument==1]
  
    methods = ['storytelling', 'vivid language', 'loaded words', 'imagery'] if args.method is None else [args.method]
    for i, row in df.iterrows():
        for method in methods:
            new_df['fallacy'].append(row['updated_label'])
            new_df['original_argument'].append(row['source_article'])
            new_df['claim'].append(row['claim'])
            new_df['method'].append(method)
            new_df['model'].append(args.model)
    labels = api(inputs=[list(new_df['original_argument']), list(new_df['method'])], temperature=0.6)
  
    new_df = pd.DataFrame(new_df)

    results = defaultdict(list)
    for i, row in new_df.iterrows():
        for k, v in labels[i].items():
            results['model'].append(row['model'])
            results['fallacy'].append(row['fallacy'])
            results['argument_ori'].append(row['original_argument'])
            results['claim'].append(row['claim'])
            results['method'].append(row['method'])
            results['emotion'].append(k)
            results['argument_gen'].append(v)
    results = pd.DataFrame(results)
    
    output_path = os.path.join(args.out_dir, args.model.split('/')[-1]+(f"_ori_{args.method}.tsv" if args.method is not None else "_ori_all.tsv"))
    
    results.to_csv(output_path, sep='\t', index=False)