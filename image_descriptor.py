import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import os
from sklearn.preprocessing import StandardScaler
# Resize images 64x128
# Convert to grayscale
# Compute x and y gradients for 8x8 cells
# Convert x and y gradients to magnitude and orientation
# Generate gradient magnitude vs orientation histograms


def hog_descriptor(image, cell_size=16, flatten=False):
    image = cv.resize(image, (64, 128))
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cells = np.reshape(image, (cell_size, cell_size, -1))
    x_grad, y_grad = compute_gradients(cells)
    mag, ang = convert2polar(x_grad, y_grad)
    hist = create_histogram(mag, ang)  # 9x128
    hist = np.reshape(hist.T, (128//cell_size, 64//cell_size, 9))
    normalized_hist = normalize_hist(hist)
    if flatten:
        hog = normalized_hist.flatten()
    else:
        hog = normalized_hist.reshape((normalized_hist.shape[0]*normalized_hist.shape[1], normalized_hist.shape[2]))
    return hog, mag


def visualize_gradients(mag):
    hog_image = np.reshape(mag.flatten(), (128, 64))
    total = np.max(hog_image)/10
    hog_image = hog_image/total
    plt.figure()
    plt.imshow(hog_image, cmap='gray')
    plt.show()
    print('a')


def compute_gradients(cell):
    x_filter = np.array([-1, 0, 1])[np.newaxis, :]
    y_filter = np.array([[1], [0], [-1]])
    x_grad = np.zeros_like(cell)
    y_grad = np.zeros_like(cell)
    for i in range(cell.shape[2]):
        x_grad[:, :, i] = ndimage.correlate(cell[:, :, i], x_filter, mode='constant', cval=0)  # zero padding around the image
        y_grad[:, :, i] = ndimage.correlate(cell[:, :, i], y_filter, mode='constant', cval=0)

    return x_grad, y_grad


def convert2polar(x_grad, y_grad):
    mag = np.sqrt(x_grad**2 + y_grad**2)
    ang = np.abs(np.arctan2(y_grad, x_grad)*180/np.pi)
    return mag, ang


def create_histogram(mag, ang):
    bins = np.arange(0, 180, 20)
    hist = np.zeros((9, mag.shape[2]))
    for bin, i in zip(bins, range(len(bins))):
        for cell in range(mag.shape[2]):
            x, y = np.where(np.logical_and(ang[:,:,cell] > bin, ang[:,:,cell] < bin+20))
            hist[i, cell] = mag[x, y, cell].sum()

    return hist


def normalize_hist(hist):
    norm_hist = np.zeros([hist.shape[0]-1, hist.shape[1]-1, 36])
    for i in range(hist.shape[0]-1):
        for j in range(hist.shape[1]-1):
            temp = hist[i:i+2, j:j+2, :]
            norm_factor = np.sqrt(np.sum(temp**2))
            if norm_factor != 0:
                norm_hist[i, j, :] = temp.flatten()/norm_factor
            else:
                norm_hist[i, j, :] = np.zeros_like(temp.flatten())
    return norm_hist


def feature_and_label_extraction_for_flat(dir, dataset, cell_size):
    feature_name = './hog_flat' + dataset + '_cell_' + str(cell_size) + '_features_scaled.npy'
    if os.path.exists(feature_name):
        hog_features = np.load(feature_name)
        image_labels = np.load('./unbalanced_' + dataset + '_class_labels.npy')
    else:

        dir = dir + dataset + '/'
        cat_list = os.listdir(dir)
        image_list = []
        count = 0
        size = 0
        for i in cat_list:
            if os.path.isdir(dir + i):
                if i == 'background_class':
                    continue
                cat = dir + i
                image_list = os.listdir(cat)
                size += len(image_list)
        set_factor = 1
        hog_features = np.zeros((size, (128//cell_size-1)*(64//cell_size-1)*36))
        image_labels = np.zeros((size, 1))
        label_no = 0
        label_pos = 0
        for i in cat_list:
            if os.path.isdir(dir + i):
                if i == 'background_class':
                    continue
                cat = dir + i
                image_list = os.listdir(cat)
                image_labels[label_pos:label_pos + set_factor*len(image_list)] = label_no
                label_no += 1
                label_pos += set_factor*len(image_list)
            for image in image_list:
                im = cv.imread(cat + '/' + image, 0)
                hog, _ = hog_descriptor(im, cell_size=cell_size, flatten=True)
                hog_features[count, :] = hog
                count += 1
        scaler = StandardScaler()
        hog_features = scaler.fit_transform(hog_features)
        np.save('./hog_flat' + dataset + '_cell_' + str(cell_size) + '_features_scaled.npy', hog_features)
        np.save('./unbalanced_' + dataset + '_class_labels.npy', image_labels)

    return hog_features, image_labels


def feature_and_label_extraction_for_nonflat(dir, dataset, cell_size):
    temp_dir = dir
    feature_name = './hog_nonflat_' + dataset + '_cell_' + str(cell_size) + '_features.npy'
    if os.path.exists(feature_name):
        hog_features = np.load(feature_name)
        new_labels = np.load('./unbalanced_no_rotation_' + dataset + '_class_labels.npy')
    else:
        dir = dir + dataset + '/'
        cat_list = os.listdir(dir)
        image_list = []
        count = 0
        size = 0
        for i in cat_list:
            if os.path.isdir(dir + i):
                if i == 'background_class':
                    continue
                cat = dir + i
                image_list = os.listdir(cat)
                size += len(image_list)
        hog_features = np.zeros((size, (128 // cell_size - 1) * (64 // cell_size - 1),  36))
        for i in cat_list:
            if os.path.isdir(dir + i):
                if i == 'background_class':
                    continue
                cat = dir + i
                image_list = os.listdir(cat)
            for image in image_list:
                im = cv.imread(cat + '/' + image, 0)
                hog, _ = hog_descriptor(im, cell_size=cell_size)
                hog_features[count, :, :] = hog
                count += 1
        np.save('./hog_nonflat_' + dataset + '_cell_' + str(cell_size) + '_features.npy', hog_features)

        _, new_labels = feature_and_label_extraction_for_flat(temp_dir, dataset, cell_size)
        np.save('./unbalanced_no_rotation_' + dataset + '_class_labels.npy', new_labels)

    return hog_features, new_labels

if __name__ == '__main__':
    dir = '../Caltech20/'
    dataset = 'testing'
    cell_size = 32
    feature_and_label_extraction_for_nonflat(dir, dataset, cell_size)