import numpy as np



def bag_of_words(data, no_cluster):
    dataset_hist = np.zeros((data.shape[0], no_cluster))
    for i in range(data.shape[0]):
        image_hist = np.unique(data[i,:], return_counts=True)
        try:
            dataset_hist[i, image_hist[0].astype(int)] = image_hist[1]
        except IndexError:
            print('a')
    return dataset_hist