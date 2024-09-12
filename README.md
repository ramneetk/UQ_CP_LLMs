
This code is built on top of code for "Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models" [arxiv](https://arxiv.org/abs/2305.19187).

# Quick Start
1. Note that to get the automatic evaluation based on GPT, you would need to update `keys.json` with your API keys first. 

2. Set the corresponding paths in `_settings.py`: these paths are where your (a) model generations (or responses), and (b) downloaded model weights of `Llama-2-13b`, and `Mistral-7B-Instruct-v0.2` (from hugging face) will be stored.

3. Install the dependies: ``` pip install -r requirements.txt ```

## Generate 20 responses from the model
Use the `llama-13b-hf`, `mistral-7b-hf` for model, and `coqa`, `triviaqa` (for exact values of hyperparameters used in the paper such as temperature, top_p etc. please read the topmost comments in pipeline/generate.py):
```
python3 -m pipeline.generate --model llama-13b-hf --dataset coqa
```
The generations will be saved in the GENERATION_FOLDER/$model_dataset_seed$/$seed$.pkl. 

%You can also download the generations from both models on both datasets from here: $path$.

## Run UQ Experiments
### Step 1: Compute scores required for UQ and GT evaluation on the generated responses
Run `dataeval/load.py` to cache down the results first (including GPT evaluations): load.py will run evaluations on all the paths specified in _settings.py:
```
python3 -m dataeval.load
```

### Step 2: Generate baseline and our results
Run `pipeline/uq_bb.py`
```
python3 -m pipeline.uq_bb --model $llama-13b-hf/mistral-7b-hf$ --dataset $coqa/triviaqa$ --cal_size $1000 for coqa/2000 for triviaqa$ --acc_name $rougeL/deberta_entailment/gpt$ --metric $auarc/auroc/rej_acc$
```

## Run Conformal Prediction Experiments (Getting accuracy and set sizes)
### Assuming Step 1 from UQ Experiments has been done, Step 2 is to run uq_bb for generating prediction sets
```
python3 -m pipeline.uq_bb --model $llama-13b-hf/mistral-7b-hf$ --dataset $coqa/triviaqa$ --cal_size $1000 for coqa/2000 for triviaqa$ --acc_name $rougeL/deberta_entailment/gpt$ --output_pred_sets True
```

## Notes on GPT evaltuaions
1. We used `gpt-4-turbo-preview` (which points to `gpt-4-0125-preview`) for GT evaluations from GPT

2. These experiments were performed using openai version 0.28. If you get this OpenAI error: "You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API." You can resolve it by installing to the old version, e.g. `pip install openai==0.28`.
