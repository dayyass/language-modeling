### About
Language Modeling is the task of evaluating the probability distribution over a sequence of words.

### Implemented models
- [x] N-gram Language Model
- [ ] RNN Language Model
- [ ] GPT Language Model

### Training
```
python train.py --path_to_data "data/arxiv.txt" --n 3 --path_to_save "models/language_model.pkl" --verbose True
```
Available argumets:
- **--path_to_data** - path to train data
- **--n** - n-gram order
- **--path_to_save** - path to save model
- **--verbose** - verbose

### Inference
```
python inference.py --path_to_model "models/language_model.pkl" --prefix "" --temperature 1.0 --max_length 100
```
Available argumets:
- **--path_to_model** - path to language model
- **--prefix** - prefix before sequence generation
- **--temperature** - sampling temperature
- **--max_length** - max number of generated words
