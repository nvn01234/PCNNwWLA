# word-level convolution
NB_FILTERS_WORD = 100  # hidden layer 1
WINDOW_SIZE_WORD = 3
WORD_EMBED_SIZE = 100
SEQUENCE_LEN = 117

# position feature
POSITION_EMBED_SIZE = 5
MAX_DISTANCE = 64
MIN_DISTANCE = -64
NB_POSITIONS = MAX_DISTANCE - MIN_DISTANCE + 1

ENTITY_LEN = 5

NB_RELATIONS = 4

DROPOUT = 0.5

WORD_REPRE_SIZE = WORD_EMBED_SIZE + 2*POSITION_EMBED_SIZE
PCNN_OUTPUT_SIZE = NB_FILTERS_WORD * 3
