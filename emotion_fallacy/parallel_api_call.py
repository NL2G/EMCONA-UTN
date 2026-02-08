from termcolor import colored
from fastllm.cache import DiskCache
#from fastllm.fastllm.core import RequestBatch, RequestManager
from fastllm.core import RequestBatch, RequestManager
#from fastllm.fastllm.providers.openai import OpenAIProvider
from fastllm.providers.openai import OpenAIProvider
from argparse import ArgumentParser
import pandas as pd
import sys
from rich import print
import re
import os
from collections import defaultdict

class APICall:
    def __init__(self, model=None, api_base=None, api_key=None, concurrency = 2, cache_dir=None, apply_template=None, parse_response=None):
        print(f"Running {model}...")
        with open(api_key, 'r') as f:
            api_key = f.read().strip()
        self.manager = RequestManager(provider=OpenAIProvider(
                api_base=api_base,
                api_key=api_key),
            caching_provider=DiskCache(
                directory=cache_dir, 
                expire=None, 
                size_limit=int(10e10), 
                cull_limit=0, 
                eviction_policy='none'
            ), concurrency=concurrency, show_progress=True, timeout=120)
        self.model = model
        self.parse_response = parse_response
        self.apply_template = apply_template
       
        
    def __call__(
        self, 
        inputs: list[list[str]],
        temperature = 0,
    ) -> dict[str, list[float]]:
        """
        Generate metric scores.

        Args:
        level: Level for which to produce scores, 'sys' or 'seg'.
        lp: Language pair, e.g. 'en-de'.
        domains: Map from domain name to [[beg, end+1], ...] segment position lists.
        docs: Map from doc name to [beg, end+1] segment positions.
        src: List of source segments.
        ref: List of reference segments.
        hyps: Map from MT system name to output segments for that system.

        Returns:
        Map from system name to scores, a list of segment-level scores if level is
        'seg', or a list containing a single score if level is 'sys'.
        """
        
        
            
        #df = pd.DataFrame({'src': src, 'hyp': hyps})
        #df = pd.DataFrame({'input1': input1, 'input2': input2})
        df = pd.DataFrame({f"inputs_{i}": inputs[i] for i in range(len(inputs))})
        #df["prompt"] = df.apply(lambda x: self.apply_template(PROMPT, x), axis=1)
        df["prompt"] = df.apply(lambda x: self.apply_template(x), axis=1)
        #print(df)
        #raise ValueError
        parse_answer = lambda x: self.parse_response(x)
        answers = self.bulk_request(df, parse_answer, max_tokens=None, temperature=temperature) # 500
        #print(type(answers))
        #print(type(answers[0]))
        #print(type(answers[0][0]))
        #print(answers[0])
        #print(len(answers))
        #print(len(answers[0]))
        return [x[0]['answer'] for x in answers]
        
    """
    def parse_mqm_answer(self, x):
        '''
        #print(x)
        try:
            x = int(x.lower().split("coherence:")[-1].strip())
            if x in [1, 2, 3, 4, 5]:
                return x
            #else:
            #    return int(re.search(pattern=r'[0-5]', string=x).group())
        except:
            print(f"cannot parse {x}")
            return None
        #raise ValueError
        '''
        return lambda x: self.parse_response(x)
    """
    
    """
    def apply_template(self, prompt_template, x):
        #prompt = prompt_template.format(Document=x['src'], Summary=x['hyp'])
        prompt = self.prompt_template.format(Document=x['src'], Summary=x['hyp'])
        prompt = [{"role": "user", "content": prompt}]
        return prompt
    """
    
    def request(self, prompts, parse_response, temperature=0, answer_id=-1, cache=None, max_tokens=None):
        ids = []
        #print(prompts[0])
        #raise ValueError
        with RequestBatch() as batch:
            #f self.models[0] !="openai/o3-mini-low":
            for i in range(len(prompts)):
                ids.append(batch.chat.completions.create(
                        model=self.model,
                        messages=prompts[i],
                        max_completion_tokens=max_tokens,
                        temperature= temperature,
                        top_p= None,
                        n= 1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                    ))
                
        answer_list = self.request_api(batch, temperature=temperature)
        

        outputs = []
        while len(outputs) == 0 or "INVALID" in outputs:
            if "INVALID" in outputs: # implementing the retry without recursion to make it somewhat parallel
                temperature += 1
                invalid_idx = [i for i, o in enumerate(outputs) if o == "INVALID"]
                invalid_prompts = [prompts[i] for i in invalid_idx]
                with RequestBatch() as invalid_batch:
                    for i in range(len(invalid_prompts)):
                        ids.append(invalid_batch.chat.completions.create(
                            model=self.model,
                            messages=invalid_prompts[i],
                            max_completion_tokens=max_tokens,
                            temperature= temperature/10,
                            top_p= 1,
                            n= 1,
                            frequency_penalty= 0,
                            presence_penalty=0,
                            stop=None,
                        ))
                
                answer_list_invalid = self.request_api(invalid_batch, temperature=temperature)
                answer_list = [answer_list[i] if o != "INVALID" else answer_list_invalid.pop(0) for i, o in enumerate(outputs)]
                outputs = []
                
            for answers, prompt in zip(answer_list, prompts):
                if len(answers) == 0: # if temp > 10, an empty list is returned
                    outputs.append([{
                            "temperature": temperature,
                            "answer_id": answer_id,
                            "answer": None,
                            "prompt": prompt,
                            "finish_reason": None,
                            "model": self.model,
                            }])
                    
                else:
                    parsed_answers = []
                    for full_answer in answers:
                        finish_reason = full_answer["finish_reason"]
                        full_answer = full_answer["answer"]

                        answer_id += 1
                        #print(full_answer)
                        #raise ValueError
                        answer = parse_response(full_answer)
                        if temperature > 0:
                            pass
                            #print(f"Answer (t={temperature}): " + colored(answer, "yellow") + " (" + colored(full_answer, "blue") + ")", file=sys.stderr)
                        if answer is None:
                            continue
                        parsed_answers.append(
                            {
                                "temperature": temperature,
                                "answer_id": answer_id,
                                "answer": answer,
                                "prompt": prompt,
                                "finish_reason": finish_reason,
                                "model": self.model
                            }
                        )

                    # there was no valid answer, increase temperature and try again
                    if len(parsed_answers) == 0:
                        outputs.append("INVALID")
                    else:
                        outputs.append(parsed_answers)
        
        return outputs

    
    def request_api(self, batch, temperature):
        if temperature > 10:
            return []
        
        i_list = [i for i in self.manager.process_batch(batch)]
        response_list = [i.response for i in i_list]
        
        print("i_list", response_list)

        answer_list = []
        for response in response_list:
            answers = []
            for choice in response.choices:
                if choice.message.content is None:
                    answer_list.append("INVALID")
                if hasattr(choice, "message"):
                    answer = choice.message.content.strip()
                else:
                    answer = choice.text.strip()
                    
                answers.append({
                    "answer": answer,
                    "finish_reason": choice.finish_reason,
                })

            if len(answers) > 1:
                # remove duplicate answers
                answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

            answer_list.append(answers)
        return answer_list
    
    def bulk_request(self, df, parse_mqm_answer, max_tokens=None, temperature=0):
        return self.request(df["prompt"].tolist(), parse_mqm_answer, max_tokens=max_tokens, temperature=temperature)


    


