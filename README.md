# PCNN with word-level attention

## Requirements

- numpy

- tensorflow

- keras

## Usage

### Input file

Default input: `input.txt`

Input file can contain multiple line. Each line must be follow this pattern

    relation e1start e1end e2start e2end word-1 word-2 ... word-n

- `relation` is integer. List of relations provided at `origin_data/relations.txt`

- `e1start`, `e1end`, `e2start`, `e2end` are starting index and ending index of two entities in the sentence

- From `word-1` to `word-n` are the preprocessed sentence (tokenized, lowercase)

Example one line of input file:

    0 0 0 4 4 lệ_quyên sinh ra tại hà_nội , trong gia_đình có 4 anh_chị_em , cô là con út

### Train

Run `initial.py` one time to preprocess data. Preprocessed data will be placed in `data` folder

    python initial.py

Run `train.py` to train model. You can edit model's parameters in `settings.py`. Train output and log will be placed in `output/train/**` folder. `**` is the timestamp, you can train model many time to get best weights.

    python train.py
    
### Train output and log

Stored in `output/train/**`:

- `result.json`: macro-averaged f1-score, max f1-score, best fold, best weights

- `fold_*` folder: contain weights and all metrics of one fold

### Test

Run `test.py` 

    python test.py example_test_input/weights.h5 example_test_input/input.txt
    
- `example_test_input/weights.h5`: path to weights of model

- `example_test_input/input.txt`: path to input file


### Test output files

Stored in `output/test/**`:

- Predictions for input file: `output.txt`

- Error sentences: `error_list.txt`, `error_predictions.txt`

