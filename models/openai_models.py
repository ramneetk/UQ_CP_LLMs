import json
import os
import time

import openai
import persist_to_disk as ptd
from openai import APIError, RateLimitError
import random
import numpy as np

import pdb

TOTAL_TOKEN = 0

with open(os.path.join(os.path.dirname(__file__), '..', 'keys.json'), 'r') as f:
    openai.api_key = json.load(f)['openai']['apiKey']


@ptd.persistf(groupby=['model'], hashsize=10000, lock_granularity='call')
def _openai_query_cached_new(prompt='Hello World', model='ada', attempt_id=0, logprobs=True, top_logprobs=1):
    #print(f"logprobs is set to {logprobs}, and top_logprobs is set to: {top_logprobs}")
    return openai.ChatCompletion.create(model=model,
                                        messages=[{"role": "user", "content": prompt}], logprobs=logprobs, top_logprobs=top_logprobs)

def retry_openai_query(prompt='Hello World', model='ada', attempt_id=0, max_tries=5,logprobs=True, top_logprobs=1):
    for i in range(max_tries):
        try:
            return _openai_query_cached_new(prompt, model, attempt_id, logprobs, top_logprobs)
        #except (RateLimitError, APIError) as e:
        except openai.OpenAIError as e:
            print("HHHHHHHELLOOOOOOOOOOOOOOO")
            print(e)
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            delay_secs = 4**i
            sleep_time = randomness_collision_avoidance+delay_secs
            print(f'*** Error {e}. Retrying for {i} time after {sleep_time} sleep time ****')
            time.sleep(sleep_time)
            if i == max_tries - 1:
                raise e
            continue

def _token_to_price(model, tokens):
    #return tokens // 1000 * {'gpt-3.5-turbo': 0.002}[model]
    return tokens // 1000 * {'gpt-4-turbo-preview': 0.002}[model]

def openai_query(prompt, model, attemptd_id, max_tries=50, verbose=False, logprobs=True, top_logprobs=5):
    global TOTAL_TOKEN
    completion = retry_openai_query(prompt, model, attemptd_id, max_tries=max_tries, logprobs=logprobs, top_logprobs=top_logprobs) # top_logprobs=n for getting top n responses

    top_outputs = []
    top_probs = []
    for itr in range(top_logprobs):
        top_outputs.append(completion.choices[0].logprobs.content[0].top_logprobs[itr].token)
        output_logprob = completion.choices[0].logprobs.content[0].top_logprobs[itr].logprob
        output_prob = np.round(np.exp(output_logprob)*100,2)
        top_probs.append(output_prob)
    
    txt_ans = completion.choices[0].message.content
    prev_milestone = _token_to_price(model, TOTAL_TOKEN) // 0.1
    TOTAL_TOKEN += completion['usage']['total_tokens']

    if (_token_to_price(model, TOTAL_TOKEN) // 0.1)  > prev_milestone:
        if verbose:
            print(f"Total Cost > $ {(_token_to_price(model, TOTAL_TOKEN) // 0.1) * 0.1:.1f}")
    return txt_ans, top_outputs, top_probs
