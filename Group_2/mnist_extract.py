from mnist import MNIST
import math
import random
import os

data_path = os.path.join(os.path.dirname(__file__), 'python-mnist','data')
mndata = MNIST(data_path)
images, labels = mndata.load_training()



def image_print(image, row_size, one_ify=True):
    row = []
    for i in range( len(image)):
        

        pixel = image[i-1]
        if one_ify:
            if pixel > 0: row.append("X")
            else: row.append("_")
        else:
            row.append(pixel)
        
        if len(row) % row_size == 0:
            print(row)
            row = []

        


def to_28_matrix(pixels):
    matrix = []
    for i in range(28):
        row = []
        for j in range(28):
            pixel = pixels[28*i + j]
            row.append(pixel)
        matrix.append(row)
    return matrix

def make_empty_matrix(width, height):
    matrix = []
    for i in range(height):
        matrix.append([0]*width)
    return matrix






def split_list(lengths, original_list):
    list_of_lists = []
    for length in lengths:
        segment = []
        for _ in range(length):
            segment.append(original_list.pop())
        list_of_lists.append(segment)
    return list_of_lists

def split_images_and_labels(images, labels):
    paired = list(zip(images,labels))
    random.shuffle(paired)
    images, labels = list(zip(*paired))

    images = list(images)
    labels = list(labels)

    num_heldout = int(0.1 * len(images))
    num_test = int(0.2 * len(images))
    num_train = len(images) - num_test - num_heldout

    heldout_X, test_X, train_X = split_list([num_heldout, num_test, num_train], images)
    heldout_y, test_y, train_y = split_list([num_heldout, num_test, num_train], labels)

    return ((heldout_X, test_X, train_X), (heldout_y, test_y, train_y))

splitted_X, splitted_y = split_images_and_labels(images, labels)
heldout_X, test_X, train_X  = splitted_X
heldout_y, test_y, train_y = splitted_y

def flatten(matrix):
    image = []
    for row in matrix:
        image.extend(row)
    return image


def compression(X):
    #[[row]] eg) [[15, 396, 0], [2,5,7]]
    #each row is decimal version of .@@..@ for each row in image
    #try this if it has good accuracy, with more time, it smells bad

    compression_X = []
    for image in X:
        threshold = sum(image)/len(image)
        matrix = to_28_matrix(image)
        for row in matrix:
            for i in range(len(row)):
                if row[i] >= threshold: row[i] = '1'
                else: row[i] = '0'

        for i in range(len(matrix)):
            binary = "".join(matrix[i])
            matrix[i] = int(binary, 2)
        
        compression_X.append(matrix)

    return compression_X

def row_density(X):
    #[row_density] eg) [[0.9,0.5,0.1],[0.1,0.5,0.0]]
    row_density_X = []

    for image in X:
        threshold = sum(image)/len(image)
        matrix = to_28_matrix(image)

        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                if matrix[r][c] >= threshold: matrix[r][c] = 1
                else: matrix[r][c] = 0

        densities = []
        col_len = len(matrix[0])
        for i in range(len(matrix)):
            density = sum(matrix[i])/col_len
            densities.append(density)

        row_density_X.append(densities)

    return row_density_X





def binary_scale(X):

    binary_scale_X = []
    for image in X:
        new_image = []
        for i in range(len(image)):
            pixel = 1 if image[i] > 100 else 0
            new_image.append(pixel)
            
        binary_scale_X.append(new_image)
    return binary_scale_X





#test stuff
'''
print("total images", len(images))
print("heldout", len(heldout_X))
print("test", len(test_X))
print("train", len(train_X))


image = train_X[0]
label = train_y[0]


image_print(image, 28)
print("should be", label)

print(len(compression([image])[0]))
print(len(row_density([image])[0]))
print(len(binary_scale([image])[0]))
'''