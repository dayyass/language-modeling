# hyper-parameters tuning
# run from language_modeling folder, not statistical_lm folder

# unigram language model
python statistical_lm/train.py --path_to_data "data/arxiv_train.txt" --n 1 --path_to_save "models/1_gram_language_model.pkl"
python statistical_lm/validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/1_gram_language_model.pkl"

# bigram language model
python statistical_lm/train.py --path_to_data "data/arxiv_train.txt" --n 2 --path_to_save "models/2_gram_language_model.pkl"
python statistical_lm/validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/2_gram_language_model.pkl"

# trigram language model
python statistical_lm/train.py --path_to_data "data/arxiv_train.txt" --n 3 --path_to_save "models/3_gram_language_model.pkl"
python statistical_lm/validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/3_gram_language_model.pkl"

# unigram language model with add-k smoothing
python statistical_lm/train.py --path_to_data "data/arxiv_train.txt" --n 1 --smoothing "add-k" --delta 0.1 --path_to_save "models/1_gram_laplace_language_model.pkl"
python statistical_lm/validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/1_gram_laplace_language_model.pkl"

# bigram language model with add-k smoothing
python statistical_lm/train.py --path_to_data "data/arxiv_train.txt" --n 2 --smoothing "add-k" --delta 0.1 --path_to_save "models/2_gram_laplace_language_model.pkl"
python statistical_lm/validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/2_gram_laplace_language_model.pkl"

# trigram language model with add-k smoothing
python statistical_lm/train.py --path_to_data "data/arxiv_train.txt" --n 3 --smoothing "add-k" --delta 0.1 --path_to_save "models/3_gram_laplace_language_model.pkl"
python statistical_lm/validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/3_gram_laplace_language_model.pkl"
