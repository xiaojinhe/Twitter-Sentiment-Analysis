import numpy as np
import pandas as pd


def load_data_and_shuffle(cleaned_pos_file, cleaned_neg_file):
    """ 
    Load cleaned_data_file and generates x and y (labels)
    Return splited data with labels. 
    """
    # Load data
    positive = pd.read_csv(cleaned_pos_file, index_col=0, encoding="utf-8")
    pos_samples = []
    for s in positive.text:
        pos_samples.append(s.strip())
    
    negative = pd.read_csv(cleaned_neg_file, index_col=0, encoding="utf-8")
    neg_samples = []
    for s in negative.text:
        neg_samples.append(s.strip())

    x = np.array(pos_samples + neg_samples)
    # Creates labels vector for both positive and negative samples
    pos_labels = [1 for _ in pos_samples]
    neg_labels = [0 for _ in neg_samples]
    y = np.array(np.concatenate([pos_labels, neg_labels], 0))

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    return x_shuffled, y_shuffled