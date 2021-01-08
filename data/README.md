### Data for Language Modeling
This is the folder to keep data.<br/>

### Data Format
Prepared text file with space separated words on each line.<br/>
Data preparation is performed on the user's side.

### Data Example
Corpora of ArXiv articles:
- [source data](https://www.dropbox.com/s/dl/99az9n1b57qkd9j/arxivData.json.tar.gz)
- [prepared data](https://drive.google.com/file/d/1dgTxPNKQG8aHDYOLeHNqYz2XRCv82liM/view?usp=sharing)

### Train/Test Split
Script for train/test split (run from data folder):
```
python train_test_split.py --path_to_data "arxiv.txt" --test_size 0.25 --random_state 42 --shuffle True --verbose True
```
Available argumets:
- **--path_to_data** - path to data
- **--test_size** - test size
- **--random_state** - random state (default: *42*)
- **--shuffle** - shuffle (default: *True*)
- **--verbose** - verbose (default: *True*)
