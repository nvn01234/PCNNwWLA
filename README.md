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

- `relation` is integer. List of relations provided at `relations.txt`

- `e1start`, `e1end`, `e2start`, `e2end` are starting index and ending index of two entities in the sentence

- From `word-1` to `word-n` are the preprocessed sentence (tokenized, lowercase)

Example one line of input file:

    0 0 0 4 4 lệ_quyên sinh ra tại hà_nội , trong gia_đình có 4 anh_chị_em , cô là con út

### Run

Run `test.py`

    python test.py


Run `test.py` with specific input file and output folder

    python test.py input.txt output
    
- `input.txt`: path to input file

- `output`: path to output folder
    

### Output files

Default output folder: `output`

Predictions for input file: `output/output.txt`

Error sentences: `output/error_list.txt`, `output/error_predictions.txt`

