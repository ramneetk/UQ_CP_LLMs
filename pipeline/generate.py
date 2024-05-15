# COQA
# python3 -m pipeline.generate --temperature 1.5 --seed 0 --top_p 0.9 --model mistral-7b-hf --model_type instruct --max_length_of_generated_sequence 50 --dataset coqa # settings similar to Colin's
# python3 -m pipeline.generate --temperature 1 --seed 10 --top_p 1 --model llama-13b-hf --model_type non_instruct --max_length_of_generated_sequence 256 --dataset coqa # orig settings
# TRIVAIQA
# python3 -m pipeline.generate --temperature 1 --seed 0 --top_p 1 --model llama-13b-hf --model_type non_instruct --max_length_of_generated_sequence 256 --dataset triviaqa # settings from Conformal Modeling Paper, seed is 10 for generations but 0 for shuffling train, val and test indices from validation dataset
# python3 -m pipeline.generate --temperature 1.5 --seed 10 --top_p 0.9 --model mistral-7b-hf --model_type instruct --max_length_of_generated_sequence 50 --dataset triviaqa # temperature settings similar to Colin's

import argparse
import glob
import json
import os

import pandas as pd
import torch
import tqdm
import transformers

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import models
import utils

import pdb

from langchain.prompts import PromptTemplate

coqa_reverse_instruct_prompt = PromptTemplate(
    input_variables=["story", "answer"],
    template = """[INST] 
        Given the following story and answer, output the question that was asked based on the sroty.
        Story = {story}
        Answer = {answer}
    [/INST]"""
)

coqa_reverse_non_instruct_prompt = PromptTemplate(
    input_variables=["story", "answer"],
    template = """
        Given the following story and answer, output the question that was asked based on the story.
        Story = {story}
        Answer = {answer}"""
)

trivia_reverse_instruct_prompt = PromptTemplate(
    input_variables=["answer"],
    template = """[INST] 
        Given the following answer, output the trivia question that was asked.
        Answer = {answer}
    [/INST]"""
)

trivia_reverse_non_instruct_prompt = PromptTemplate(
    input_variables=["answer"],
    template = """
        Given the following answer, output the trivia question that was asked.
        answer = {answer}"""
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf') # Mistral-7B-Instruct-v0.2 for mistral
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=20)
parser.add_argument('--temperature', type=float, default='1.0') # 1 for COQA on LLAMA-2-13B, 1.5 for COQA on Mistral-7B-Instruct-v0.2
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=1.0) # 1 for COQA on LLAMA-2-13B, 0.9 for COQA on Mistral-7B-Instruct-v0.2
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--seed', type=int, default=10) # 10 for COQA on LLAMA-2-13B, 0 for COQA on Mistral-7B-Instruct-v0.2
parser.add_argument('--model_type', type=str, default='non_instruct') # non_instruct (llama) vs instruct (mistral)
parser.add_argument('--max_length_of_generated_sequence', type=int, default=256) # 256 for llama, 50 for mistral
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--prompt_type', type=str, default='direct') # direct for generating answers and reverse for generating question for an answer
###### ALSO change "max_length_of_generated_sequence": 256 for COQA on LLAMA-2-13B, and 50 for COQA on Mistral-7B-Instruct-v0.2

args = parser.parse_args()

############ for TRIVIAQA dataset from Conformal Language Modeling paper ###############################
if args.dataset == 'triviaqa' and args.model == 'llama-13b-hf':
    # generations file path
    triviaQA_generations_path = os.path.join(_settings.GENERATION_FOLDER, f'{args.model}_{args.dataset}_{args.seed}')
    triviaQA_generations_file = triviaQA_generations_path+'/validation.jsonl'
    # loading the generations file
    import json
    import numpy as np
    import pickle
    with open(triviaQA_generations_file, 'r') as json_file:
        json_list = list(json_file)
    # getting indices of train, val and test set
    utils.seed_everything(args.seed)
    shuffle = np.random.permutation(len(json_list))
    assert np.all(shuffle[0:10] == [10458,  9792, 11415,  2043,  1243,  3569,  8827, 14780, 13114,  9760]) # sanity check on indices from conformal modeling paper
    num_train=2000
    num_val=2000
    splits = {
        'train': shuffle[:num_train],
        'val': shuffle[num_train:num_train + num_val],
        'test': shuffle[num_train + num_val:],
    }
    with open(f'{triviaQA_generations_path}/cal_test_info/split_indices.pkl', 'wb') as f:
        pickle.dump(splits, f)
    # with open(f'{triviaQA_generations_path}/split_indices.pkl', 'rb') as f: splits = pickle.load(f)
###############################################################################################
_UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'coqa':
        return coqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset

def get_generation_config(input_ids, tokenizer, data_name):
    #assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = args.max_length_of_generated_sequence # 50 for mistral, 256 for llama
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    if data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config

@torch.no_grad()
def get_generations(model_name:str, args, seed=10, old_sequences=None, max_num_gen_once=4):
    device = args.device

    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer, model_type=args.model_type) # prompt will depend on the type of model: instruct (mistral) vs non-instruct (llama))
    # prompt type can be either direct for generating answers or reverse for generating question for an answer
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    if args.prompt_type == 'direct': # orig_settings: for generating answers to the dataset questions
        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            if batch['id'][0] in old_sequences:
                sequences.append(old_sequences[batch['id'][0]])
                continue

            input_ids = batch['input_ids'].to(device) # input_ids = tokenized prompt
            input_length = input_ids.shape[1]
            assert len(input_ids.shape) == 2 # asserting that the shape of input_ids is in a batch
            generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
            generation_config = transformers.GenerationConfig(**generation_config)
            if args.decoding_method == 'beam_search':
                raise NotImplementedError()
            elif args.decoding_method == 'greedy':
                if args.dataset != 'triviaqa': # orig settings
                    most_likely_generations = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                                            num_beams=1,
                                                            do_sample=False,
                                                            generation_config=generation_config).cpu()[0, input_length:]
                else:
                    most_likely_generations = [] # we do not anyways require most likely generations
            generations = []
            num_gens = args.num_generations_per_prompt
            
            if args.dataset != 'triviaqa' or args.model != 'llama-13b-hf': # orig settings
                while num_gens > 0:
                    _ =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                            num_beams=1, num_return_sequences=min(max_num_gen_once, num_gens),
                                            do_sample=True, top_p=args.top_p, top_k=args.top_k,
                                            temperature=args.temperature, generation_config=generation_config,
                                            )
                    generations.append(_[:, input_length:].cpu())
                    num_gens -= len(_)

                generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id) # shape = torch.Size([5, 4, 9])
                generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
                generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
            
            else:
                # read the generations from file
                result = json.loads(json_list[batch_idx])
                generations = []
                generated_texts = []
                for gen_idx in range(args.num_generations_per_prompt):
                    generations.append(result['generations'][gen_idx]['tokens'])
                    generated_texts.append(result['generations'][gen_idx]['decoded'])
                generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
                generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]

            # remember the data
            curr_seq = dict(
                prompt=input_ids.cpu()[0],
                id=batch['id'][0],
                question=batch['question'][0],
                answer=batch['answer'][0],
                additional_answers=[],
            )
            curr_seq.update(
                dict(
                    most_likely_generation_ids = most_likely_generations,
                    generations_ids=generations,
                )
            )
            curr_seq.update(
                dict(
                    most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                    generations=generated_texts,
                )
            )
            if args.dataset == 'coqa':
                curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]

        sequences.append(curr_seq)
    
    elif args.prompt_type == 'reverse': # asking model to output questions for the generated answers
        # if coqa, we also need story for the reverse prompt
        stories = []
        if args.dataset == 'coqa':
            for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                stories.append(batch['story'])
        
        # need to read the answer first for generating the reverse prompt
        from _settings import GEN_PATHS
        import pickle
        generations_file_path = GEN_PATHS[args.dataset][args.model]
        generations_file = open(generations_file_path, 'rb')
        generated_file_contents = pickle.load(generations_file)

        for idx in range(len(generated_file_contents)): # for each question in the dataset
            saved_generations_content = generated_file_contents[idx] # generated_answers is a dictionary with these keys: 'prompt', 'id', 'question', 'answer', 'additional_answers', 'most_likely_generation_ids', 'generations_ids', 'most_likely_generation', 'generations'
            id = saved_generations_content['id']
            question = saved_generations_content['question']
            answer = saved_generations_content['answer']
            additional_answers = saved_generations_content['additional_answers']
            
            most_likely_generations = [] # we do not require most likely generations but save it to be compatible with the original code (Jimeng's data generation code)
            generations = []
            all_input_ids = []

            for res_idx in range(args.num_generations_per_prompt): # iterating over each generated answer for the question
                generated_answer = saved_generations_content['generations'][res_idx] 
                if args.prompt_type == 'instruct':
                    if args.dataset == 'triviaqa':
                        input_prompt = trivia_reverse_instruct_prompt.format(answer=generated_answer)
                    elif args.dataset == 'coqa':
                        input_prompt = coqa_reverse_instruct_prompt.format(story=stories[idx][0],answer=generated_answer)
                else: # non_instruct prompt for non-intruct models
                    if args.dataset == 'triviaqa':
                        input_prompt = trivia_reverse_non_instruct_prompt.format(answer=generated_answer)
                    elif args.dataset == 'coqa':
                        input_prompt = coqa_reverse_non_instruct_prompt.format(story=stories[idx][0],answer=generated_answer)
                
                # generating tokens for prompt
                input_ids_attention_mask = tokenizer(input_prompt, truncation=False, padding=False)
                input_ids = [input_ids_attention_mask['input_ids']] # input to model is in a batch, here batch length = 1
                input_ids = torch.tensor(input_ids)
                input_ids = input_ids.to(device)
                attention_mask = [input_ids_attention_mask['attention_mask']] # input to model is in a batch, here batch length = 1
                attention_mask = torch.tensor(attention_mask)
                attention_mask = attention_mask.to(device)
                
                assert len(input_ids.shape) == 2 # input_ids is in a batch
                input_length = input_ids.shape[1]
                generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
                generation_config = transformers.GenerationConfig(**generation_config)

                _ = model.generate(input_ids, attention_mask=attention_mask, num_beams=1, num_return_sequences=1, do_sample=True, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, generation_config=generation_config) # num_return_sequences will be 1 here as it is a different prompt for each generated answer unlike the original settings
                generations.append(_[:, input_length:].cpu())
                all_input_ids.append(input_ids.cpu()[0])

        
            generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
            generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
            generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
            pdb.set_trace()

            # remember the data
            curr_seq = dict(
                prompt=all_input_ids, # this is different from direct prompt as input_ids is same for all generations but here input_id (or tokenized prompt) will be different for each reverse prompt with the different generated answer. So it will be a list of 20 prompts here 
                id=id, # id of the original question from the dataset
                question=question, # original question from the dataset
                answer=answer, # original GT answer from the dataset
                additional_answers=additional_answers, # original additional answers from the dataset
            )
            curr_seq.update(
                dict(
                    most_likely_generation_ids = most_likely_generations,
                    generations_ids=generations,
                )
            )
            curr_seq.update(
                dict(
                    most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                    generations=generated_texts,
                )
            )
            sequences.append(curr_seq)

    return sequences

# not modifying this for triviaQA, as we are not running tests with GPT models
def get_generations_bb(model_name:str, args, seed=10, old_sequences=None, task_runner:utils.TaskPartitioner=None):
    # ='gpt-4-turbo-preview'
    dataset = get_dataset_fn(args.dataset)(_UNUSED_TOKENIZER, model_type=args.model_type) # prompt will depend on the type of model: instruct (mistral) vs non-instruct (llama))
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}
    sequences = []
    print("Total datapoints: ", len(dataloader))
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['id'][0] in old_sequences:
            sequences.append(old_sequences[batch['id'][0]])
            continue
        generated_texts = []
        for _ in range(args.num_generations_per_prompt):
            if task_runner is None:
                generated_texts.append(models.openai_query(batch['prompt'][0], model=model_name, attemptd_id=_, max_tries=50))
            else:
                task_runner.add_task(models.openai_query, batch['prompt'][0], model=model_name, attemptd_id=_, max_tries=50)
        if task_runner is not None:
            continue
        curr_seq = dict(
                prompt=batch['prompt'][0],
                id=batch['id'][0], #NOTE: This changed
                question=batch['question'][0],
                answer=batch['answer'][0], #NOTE: This changed
                additional_answers=[],
        )
        curr_seq.update(
                dict(
                    generations=generated_texts,
                )
            )

        if args.dataset == 'coqa':
            curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]
        sequences.append(curr_seq)
    return task_runner or sequences

def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.seed}')
        if args.prompt_type == 'reverse':
            cache_dir = cache_dir+'/reverse_prompt_results'
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)


        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    #if model_name == 'gpt-3.5-turbo':
    if model_name == 'gpt-4-turbo-preview':
        task_runner = None if parallel is None else utils.TaskPartitioner()
        sequences = get_generations_bb(model_name, args, seed=args.seed, old_sequences=old_sequences, task_runner=task_runner)
        if task_runner is not None:
            # caching down the results but will need to run again without parallelisation to save the result in a pkl file
            return task_runner.run_multi_process(parallel)
        print(f'Writing {len(sequences)} generations to {cache_dir}...')
        pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
        return
    else:
        sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences)
        print(f'Writing {len(sequences)} generations to {cache_dir}...')
        pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
        return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess)
