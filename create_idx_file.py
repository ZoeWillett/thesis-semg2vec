import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# windows function from https://github.com/MansoorehMontazerin/Vision-Transformer_EMG
def windows(end, window_size, skip_step):
    """
    It creates a generator yielding windows with the length of the window_size.
    It starts to segment a sequence from the sample's start to its end.
 
    >>> gen = windows(1000, 200, 20)
    >>> next(gen)
    (0, 200)
    
    >>> next(gen)
    (20, 220)
    """
    start= 0
    while (start + window_size) <= end:
        yield start, start + window_size
        start += skip_step

def load_subjects(idx):
    samples = []
    labels = []
    for i in idx:
        print("loading subj ", i, "...")
        with open ('preprocessed/subj' + str(i) + '_samples', 'rb') as fp:
            data = pickle.load(fp)
        samples = samples + data
        label = np.load("preprocessed/subj" + str(i) + "_labels.npy")
        if i == 1:
            labels = label
        else:
            labels = np.concatenate((labels, label))
    labels_df = pd.DataFrame(labels, columns=['subject', 'gesture', 'repetition'], dtype='int')
    labels_df['sample'] = range(len(labels_df))
    return samples, labels_df


def make_idx_list(samples, window_size, skip_step):
    sample_list = []
    window_start = []
    window_stop = []
    
    for s, sample in samples:
        for start, stop in windows(len(sample), window_size, skip_step):
            sample_list.append(s)
            window_start.append(start)
            window_stop.append(stop)
    
    index_df = pd.DataFrame(
        {'sample': sample_list,
         'window_start': window_start,
         'window_stop': window_stop
        })
    return index_df


if __name__ == "__main__":
    
    print("loading subjects into one file...")
    idxs = [id for id in range(1,21)]
    samples, labels_df = load_subjects(idxs) 

    '''
    print("saving labels as labels.csv...")
    labels_df.to_csv('preprocessed/new_split/all_labels.csv', index=False) # ['subject', 'gesture', 'repetition', 'sample']

    print("train test val split...")
    # train test val split
    train, test = train_test_split(labels_df, test_size=0.2, shuffle=True)
    # train validation split
    train, val = train_test_split(train, test_size=0.25, shuffle=True)
    # 60% train, 20% validation and 20% test

    print("saving train test val labels...")
    train.to_csv('preprocessed/new_split/train_labels.csv', index=False)
    test.to_csv('preprocessed/new_split/test_labels.csv', index=False)
    val.to_csv('preprocessed/new_split/val_labels.csv', index=False)
    '''
    train = pd.read_csv('preprocessed/new_split/train_labels.csv')
    test = pd.read_csv('preprocessed/new_split/test_labels.csv')
    val = pd.read_csv('preprocessed/new_split/val_labels.csv')

    samples_df = pd.DataFrame({'samples': samples})
    
    print("creating index lists...")
    samples_train = [(idx, sample) for idx, sample in enumerate(samples) if idx in train['sample']]
    samples_train = samples_df.loc[train['sample']]
    samples_train = [(idx, row['samples']) for idx, row in samples_train.iterrows()]
    idx_df_train = make_idx_list(samples_train, 64, 32) # paper: 64, 32 # default: 128, 32

    samples_test = [(idx, sample) for idx, sample in enumerate(samples) if idx in test['sample']]
    samples_test = samples_df.loc[test['sample']]
    samples_test = [(idx, row['samples']) for idx, row in samples_test.iterrows()]
    idx_df_test = make_idx_list(samples_test, 64, 32) # paper: 64, 32 # default: 128, 32

    samples_val = [(idx, sample) for idx, sample in enumerate(samples) if idx in val['sample']]
    samples_val = samples_df.loc[val['sample']]
    samples_val = [(idx, row['samples']) for idx, row in samples_val.iterrows()]
    idx_df_val = make_idx_list(samples_val, 64, 32) # paper: 64, 32 # default: 128, 32

    print("saving idx lists...")
    idx_df_train.to_csv('preprocessed/new_split/idx_list_train_paper.csv', index=False)
    idx_df_test.to_csv('preprocessed/new_split/idx_list_test_paper.csv', index=False)
    idx_df_val.to_csv('preprocessed/new_split/idx_list_val_paper.csv', index=False)

    print("done!")
    
'''
    print("creating index list...")
    idx_df = make_idx_list(samples, 128, 32) # paper: 64, 32

    print("saving idx list as idx_list.csv...")
    idx_df.to_csv('preprocessed/idx_list.csv', index=False)

    print("train test val split...")
    # train test val split
    train, test = train_test_split(idx_df, test_size=0.2, shuffle=True)
    # train validation split
    train, val = train_test_split(train, test_size=0.25, shuffle=True)
    # 60% train, 20% validation and 20% test

    print("saving train test val...")
    train.to_csv('preprocessed/split/train.csv', index=False)
    test.to_csv('preprocessed/split/test.csv', index=False)
    val.to_csv('preprocessed/split/val.csv', index=False)

    print("done!")
'''
