### About
Language Modeling is the task of evaluating the probability distribution over a sequence of words.

### Implemented models
- [x] N-gram Language Model
- [ ] RNN Language Model
- [ ] GPT Language Model

### Usage
Training:
```
python train.py --path "data/arxiv.txt" --n 3 --path_to_save "models/language_model.pkl" --verbose True
```
Available argumets:
- **--path** - path to train data
- **--n** - n-gram order
- **--path_to_save** - path to save model
- **--verbose** - verbose
