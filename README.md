
Built on top of code for "Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models" [arxiv](https://arxiv.org/abs/2305.19187).

# Quick Start
Note that to get the automatic evaluation based on GPT, you would need to update `keys.json` with your API keys first.
First, set the corresponding paths in `_settings.py`.

## Generate 20 responses from the model
Use the `llama-13b-hf`, `mistral-7b-hf` or `llama-3-8b` for model, and `coqa`, `triviaqa` or $new dataset$ for the dataset  below. (You need to download the LLaMA weights first).
```
python -m pipeline.generate --model llama-13b-hf --dataset coqa
```
Update `GEN_PATHS` in `_settings.py` for next steps.

## Run UQ Experiments
### Step 1: Compute scores required for UQ and GT evaluation on the generated responses
Run `dataeval/load.py` to cache down the results first (including GPT evaluations): load.py will run evaluations on all the paths specified in _settings.py, so you can start with just llama-3-8b on coqa
```
python -m dataeval.load
```
### Step 2: Generate Final Results
Run `pipeline/uq_bb.py` with three metrics: auarc, auroc, and rej_acc by changing 'auarc' in line 829 with the desired metric
You need to also need replace ['coqa']['llama-13b'] with ['coqa']['llama-3-8b'] for fetching the generations' path from _settings.py before running uq_bb.py in line 794 

## Note
As many may have noticed, `gpt-4-turbo-preview` (which points to `gpt-4-0125-preview`) for GT evaluations
