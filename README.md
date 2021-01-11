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
python statistical_lm/train.py --path_to_data "data/arxiv_train.txt" --n 3 --path_to_save "models/3_gram_language_model.pkl"
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
python statistical_lm/validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/3_gram_language_model.pkl"
```
Required arguments:
- **--path_to_data** - path to validation data
- **--path_to_model** - path to language model

Optional arguments:
- **--verbose** - verbose (default: *True*)

#### Inference
Script for generation new sequences using statistical language models:
```
python statistical_lm/inference.py --path_to_model "models/3_gram_language_model.pkl" --prefix "artificial" --temperature 0.5
```
Required arguments:
- **--path_to_model** - path to language model

Optional arguments:
- **--prefix** - prefix before sequence generation (default: *""*)
- **--temperature** - sampling temperature, if temperature == 0.0, always takes most likely token - greedy decoding (default: *0.0*)
- **--max_length** - max number of generated words (default: *100*)
- **--seed** - random seed (default: *42*)

Command output with 3-gram language model trained on [*arxiv.txt*](data/README.md) with prefix "*artificial*" and temperature 0.5:
```
artificial neural network ( cnn ) architectures on h2o platform for real - world applications . <EOS>
```

### Neural Language Modeling
#### Training
Script for training RNN language models:
```
python neural_lm/train.py --path_to_data "data/arxiv_train.txt" --path_to_save_folder "models/rnn_language_model" --n_epoch 5 --max_length 512 --batch_size 128 --embedding_dim 64 --rnn_hidden_size 256
```
Required arguments:
- **--path_to_data** - path to train data
- **--n_epoch** - number of epochs
- **--batch_size** - dataloader batch_size
- **--embedding_dim** - embedding dimension
- **--rnn_hidden_size** - LSTM hidden size

Optional arguments:
- **--path_to_save_folder** - path to save folder (default: *"models/rnn_language_model"*)
- **--max_length** - max sentence length (chars) (default: *None*)
- **--shuffle** - dataloader shuffle (default: *True*)
- **--rnn_num_layers** - number of LSTM layers (default: *1*)
- **--rnn_dropout** - LSTM dropout (default: *0.0*)
- **--train_eval_freq** - evaluation frequency (number of batches) (default: *50*)
- **--clip_grad_norm** - max_norm parameter in clip_grad_norm (default: *1.0*)
- **--seed** - random seed (default: *42*)
- **--device** - torch device (available: "cpu", "cuda") (default: *"cuda"*)
- **--verbose** - verbose (default: *True*)

#### Validation
Script for validation RNN language models using perplexity:
```
python neural_lm/validate.py --path_to_data "data/arxiv_test.txt" --path_to_model_folder "models/rnn_language_model" --max_length 512
```
Required arguments:
- **--path_to_data** - path to validation data
- **--path_to_model** - path to language model

Optional arguments:
- **--max_length** - max sentence length (chars) (default: *None*)
- **--seed** - random seed (default: *42*)
- **--device** - torch device (available: "cpu", "cuda") (default: *"cuda"*)
- **--verbose** - verbose (default: *True*)

#### Inference
Script for generation new sequences using RNN language models:
```
python neural_lm/inference.py --path_to_model_folder "models/rnn_language_model" --prefix "artificial" --temperature 0.5
```
Required arguments:
- **--path_to_model_folder** - path to language model folder

Optional arguments:
- **--prefix** - prefix before sequence generation (default: *""*)
- **--temperature** - sampling temperature, if temperature == 0.0, always takes most likely token - greedy decoding (default: *0.0*)
- **--max_length** - max number of generated tokens (chars) (default: *100*)
- **--seed** - random seed (default: *42*)
- **--device** - torch device (available: "cpu", "cuda") (default: *"cuda"*)

Command output with RNN language model trained on [*arxiv.txt*](data/README.md) with prefix "*artificial*" and temperature 0.5:
```
artificial visual information of the number , using an intervidence for detection for order to the recognition
```

### Data Format
More about it [here](data/README.md).

### Models
List of implemented models:
- [x] [N-gram Language Model](https://github.com/dayyass/language_modeling/blob/b962edac04dfe10a3f87dfa16d4d37508af6d5de/model.py#L57)
- [x] [RNN Language Model](https://github.com/dayyass/language_modeling/blob/407d02b79d6d7fd614dc7c5fd235ad269cddcb2d/neural_lm/model.py#L6) (char-based)
- [ ] GPT Language Model

### Smoothing (only for N-gram Language Models)
- [x] no smoothing
- [x] add-k / Laplace smoothing
- [ ] interpolation smoothing
- [ ] back-off / Katz smoothing
- [ ] Kneser-Ney smoothing
