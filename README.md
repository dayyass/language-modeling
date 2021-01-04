### About
Language Modeling is the task of evaluating the probability distribution over a sequence of words.

### Implemented models
- [x] N-gram Language Model
- [ ] RNN Language Model
- [ ] GPT Language Model

### Data format
More about it [here](data/README.md).

### Training
Script for training Language Models.
```
python train.py --path_to_data "data/arxiv_train.txt" --n 3 --path_to_save "models/3_gram_language_model.pkl" --verbose True
```
Available argumets:
- **--path_to_data** - path to train data
- **--n** - n-gram order
- **--path_to_save** - path to save model (default: *"models/language_model.pkl"*)
- **--verbose** - verbose (default: *True*)

### Validation
Script for validation Language Models using Perplexity.
```
python validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/3_gram_language_model.pkl" --verbose True
```
Available argumets:
- **--path_to_data** - path to train data
- **--path_to_model** - path to language model
- **--verbose** - verbose (default: *True*)

### Inference
Script for generation new sequences using Language Models.
```
python inference.py --path_to_model "models/3_gram_language_model.pkl" --prefix "artificial" --temperature 0.0 --max_length 100
```
Available argumets:
- **--path_to_model** - path to language model
- **--prefix** - prefix before sequence generation (default: *""*)
- **--temperature** - sampling temperature, if temperature == 0.0, always takes most likely token - greedy decoding (default: *0.0*)
- **--max_length** - max number of generated words (default: *100*)

Command output with 3-gram Language Model trained on [*arxiv.txt*](data/README.md) with prefix "*artificial*" and greedy decoding (temperature == 0.0):
```
artificial intelligence ( ai ) is a challenging task . <EOS>
```
