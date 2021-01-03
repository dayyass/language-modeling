### About
Language Modeling is the task of evaluating the probability distribution over a sequence of words.

### Implemented models
- [x] N-gram Language Model
- [ ] RNN Language Model
- [ ] GPT Language Model

### Usage
####Training:
```
python train.py --path "data/arxiv.txt" --n 3
```
Available argumets:
- *--path*, type=str, required=True, help="path to train data"
- *--n*, type=int, required=True, help="n-gram order"
- *--path_to_save*, type=str, required=False, default="models/language_model.pkl", help="path to save model"
- *--verbose*, type=bool, required=False, default=True, help="verbose"
