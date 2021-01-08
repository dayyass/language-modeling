# hyper-parameters tuning

# unigram language model
python train.py --path_to_data "data/arxiv_train.txt" --n 1 --path_to_save "models/1_gram_language_model.pkl" --verbose True
python validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/1_gram_language_model.pkl" --verbose True

# bigram language model
python train.py --path_to_data "data/arxiv_train.txt" --n 2 --path_to_save "models/2_gram_language_model.pkl" --verbose True
python validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/2_gram_language_model.pkl" --verbose True

# trigram language model
python train.py --path_to_data "data/arxiv_train.txt" --n 3 --path_to_save "models/3_gram_language_model.pkl" --verbose True
python validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/3_gram_language_model.pkl" --verbose True

# unigram language model with add-k smoothing
python train.py --path_to_data "data/arxiv_train.txt" --n 1 --smoothing "add-k" --delta 0.1 --path_to_save "models/1_gram_laplace_language_model.pkl" --verbose True
python validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/1_gram_laplace_language_model.pkl" --verbose True

# bigram language model with add-k smoothing
python train.py --path_to_data "data/arxiv_train.txt" --n 2 --smoothing "add-k" --delta 0.1 --path_to_save "models/2_gram_laplace_language_model.pkl" --verbose True
python validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/2_gram_laplace_language_model.pkl" --verbose True

# trigram language model with add-k smoothing
python train.py --path_to_data "data/arxiv_train.txt" --n 3 --smoothing "add-k" --delta 0.1 --path_to_save "models/3_gram_laplace_language_model.pkl" --verbose True
python validate.py --path_to_data "data/arxiv_test.txt" --path_to_model "models/3_gram_laplace_language_model.pkl" --verbose True
