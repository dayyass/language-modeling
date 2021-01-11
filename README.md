### About
Pipeline for training Language Models using PyTorch.<br/>
Inspired by Yandex Data School [NLP Course](https://github.com/yandexdataschool/nlp_course) ([week03](https://github.com/yandexdataschool/nlp_course/tree/2020/week03_lm): Language Modeling)<br/>

### Usage
First, install dependencies:
```
# clone repo
git clone https://github.com/dayyass/language_modeling.git

# install dependencies
cd language_modeling
pip install -r requirements.txt
```

### Statistical Language Modeling
#### Training
Script for training statistical language models:
```
python statistical_lm/train.py --path_to_data "data/arxiv_train.txt" --n 3 --path_to_save "models/3_gram_language_model.pkl" --verbose True
```
Required arguments:
- **--path_to_data** - path to train data
- **--n** - n-gram order

Optional arguments:
- **--smoothing** - smoothing method (available: None, "add-k") (default: *None*)
- **--delta** - smoothing additive parameter (only for add-k smoothing) (default: 1.0)
- **--path_to_save** - path to save model (default: *"models/language_model.pkl"*)
- **--verbose** - verbose (default: *True*)

#### Validation
Script for validation statistical language models using perplexity:
```
python statistical_lm/validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/3_gram_language_model.pkl" --verbose True
```
Required arguments:
- **--path_to_data** - path to train data
- **--path_to_model** - path to language model

Optional arguments:
- **--verbose** - verbose (default: *True*)

#### Inference
Script for generation new sequences using statistical language models:
```
python statistical_lm/inference.py --path_to_model "models/3_gram_language_model.pkl" --prefix "artificial" --temperature 0.0 --max_length 100
```
Required arguments:
- **--path_to_model** - path to language model

Optional arguments:
- **--prefix** - prefix before sequence generation (default: *""*)
- **--temperature** - sampling temperature, if temperature == 0.0, always takes most likely token - greedy decoding (default: *0.0*)
- **--max_length** - max number of generated words (default: *100*)

Command output with 3-gram language model trained on [*arxiv.txt*](data/README.md) with prefix "*artificial*" and greedy decoding (temperature == 0.0):
```
artificial intelligence ( ai ) is a challenging task . <EOS>
```

### Neural Language Modeling
#### Training
Script for training RNN language models:
```
python neural_lm/train.py --path_to_data "data/arxiv_test.txt" --path_to_save_folder "models/rnn_language_model" --batch_size 128 --embedding_dim 64 --rnn_hidden_size 256
```
Required arguments:
- **--path_to_data** - path to train data
- **--batch_size** - dataloader batch_size
- **--embedding_dim** - embedding dimension
- **--rnn_hidden_size** - LSTM hidden size

Optional arguments:
- **--path_to_save_folder** - path to save folder (default: *"models/rnn_language_model"*)
- **--max_len** - max sentence length (chars) (default: *None*)
- **--shuffle** - dataloader shuffle (default: *True*)
- **--rnn_num_layers** - number of LSTM layers (default: *1*)
- **--rnn_dropout** - LSTM dropout (default: *0.0*)
- **--train_eval_freq** - evaluation frequency (number of batches) (default: *50*)
- **--clip_grad_norm** - max_norm parameter in clip_grad_norm (default: *1.0*)
- **--seed** - random seed (default: *42*)
- **--device** - torch device (available: "cpu", "cuda") (default: *"cuda"*)
- **--verbose** - verbose (default: *True*)

#### Inference
Script for generation new sequences using neural language models:
```
python neural_lm/inference.py --path_to_model_folder "models/rnn_language_model" --prefix "artificial" --temperature 0.0 --max_length 100
```
Required arguments:
- **--path_to_model_folder** - path to language model folder

Optional arguments:
- **--prefix** - prefix before sequence generation (default: *""*)
- **--temperature** - sampling temperature, if temperature == 0.0, always takes most likely token - greedy decoding (default: *0.0*)
- **--max_length** - max number of generated tokens (chars) (default: *100*)
- **--seed** - random seed (default: *42*)
- **--device** - torch device (available: "cpu", "cuda") (default: *"cuda"*)

# TODO: fix
Command output with RNN language model trained on [*arxiv.txt*](data/README.md) with prefix "*artificial*" and greedy decoding (temperature == 0.0):
```
artificial intelligence ( ai ) is a challenging task . <EOS>
```

### Data Format
More about it [here](data/README.md).

### Models
List of implemented models:
- [x] [N-gram Language Model](https://github.com/dayyass/language_modeling/blob/b962edac04dfe10a3f87dfa16d4d37508af6d5de/model.py#L57)
- [x] RNN Language Model (char-based)  # TODO
- [ ] GPT Language Model

### Smoothing (only for N-gram Language Models)
- [x] no smoothing
- [x] add-k / Laplace smoothing
- [ ] interpolation smoothing
- [ ] back-off / Katz smoothing
- [ ] Kneser-Ney smoothing
