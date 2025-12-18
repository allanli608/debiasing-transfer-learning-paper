1. `conda env create -f environment_local.yml`
2. `conda env update -f environment_local.yml --prune`
3. `conda activate deeplearning-local`
4. `python ./src/data_collection/wnc_download.py`
5. run `wnc_download.py` and then `preprocess.py`


what we have so far:
- a model that converts chinese biased -> chinese neutral
- a model that converst english biased -> english neutral

what we need:
- 50 good quality chinese biased and neutral pairs
- a pipeline to wrap a model to do chinese biased -> english biased -> model -> english unbiased -> chinese unbiased
- zero shot on englih biased -> chinese neutral
- chinese biased -> chinese netural

TESTING HARNESS 
- bert model to detect bias or not bias
- perplexity or something
- semantic check?

