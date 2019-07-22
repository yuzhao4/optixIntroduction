"""
================================
Recognizing hand-written digits
================================

An example showing how the scikit-learn can be used to recognize images of
hand-written digits.

This example is commented in the
:ref:`tutorial section of the user manual <introduction>`.

"""

import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from sklearn import datasets, svm, metrics
import cv2
import glob
import imutils
from scipy.ndimage.filters import gaussian_filter

print(__doc__)

'''
@ This function is used to con
@ image_names   This is the list of image names
@ threshold     Threshold used to binary image to segement
@ return        The list of particles centroids of each image in image_names
'''
def get_contour_centroids(image_names, threshold):
    size = len(image_names)
    print('size is: ', size)
    image_coords_calculated = []

    for i in range(size):
        img = cv2.imread(image_names[i],0)
        thresh = cv2.threshold(img, threshold, 255, 0)[1]
##        cv2.imshow('thres', thresh)
##        cv2.waitKey(0)
##        print("thresh shape is: ", thresh)
        coords = np.empty((0,2))

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coords = np.append(coords, np.array([cX,cY]).reshape((1,2)), axis=0)
        coords.sort(axis=0)
        real_coords = np.loadtxt(single_coord_names[i],delimiter=' ')
        real_coords.sort(axis=0)
        image_coords_calculated.append([coords,real_coords])
    return image_coords_calculated

'''
@ This function is used to move a single particle and track the (probability, x, y)
@ img                  The img with particles with shape((x,x))
@ centroid             The calculated centroid passed to shake
@ svm1                 The svm model used to predict its one or two particle
@ svm2                 The svm model to calculate probability with double particle data
@ svm3                 The svm model to calculate probability with single && double particle data all labeled one
@ radius               The radius to move around the centroid
@ part_width           The particle image width
@ threshold            The threshold to determine to return one or two probability 
@ return               The moved single particle image with shape((1, -1))
'''
def move_particle(image, centroid, svm1, svm2, svm3, radius, part_width, threshold):
    img = image.copy()
    img = np.pad(img, pad_width=((radius,radius),(radius,radius)),mode='median')
    particles = np.empty((0, part_width * part_width))
    # can I replace following loop to increase the speed??

    x = centroid[0]
    y = centroid[1]

    label = 0

    if radius == 0:
        print("radius is zero")
        exit()
        
    left = x
    right = left + part_width
    top = y
    bottom = top + part_width
    segment = img[top:bottom, left:right]
##    cv2.imshow("wtf", segment)
##    print("------------")
##    print(left, right, top, bottom)
##    cv2.waitKey(0)
    segment = segment.reshape((1,-1))        

    label = svm1.predict(segment)
##    print("label is: ", label)
##    print("------------")

    if label == 1:
        for i in range(2 * radius + 1):
            for j in range(2 * radius + 1):
                left = x + i - int(part_width / 2)
                right = left + part_width
                top = y + j - int(part_width / 2)
                bottom = top + part_width
                segment = img[top:bottom, left:right]
                
                if segment.shape[0] == part_width and segment.shape[1] == part_width:
                    segment = segment.reshape((1,-1))
                    particles = np.append(particles, segment, axis=0)
                else:
                    particles = np.append(particles, np.zeros((1, part_width * part_width))-1,axis=0)
        if particles.shape[0] != 0:        
            pred_prob = svm3.predict_proba(particles)
            # here assume the correct label of pred_prob is at 1 along axis 1
            best_fit_index = np.argmax(pred_prob[:,1])
            best_fit_prob = pred_prob[best_fit_index,1]
            bxx = int(best_fit_index / (2 * radius + 1))
            byy = int(best_fit_index % (2 * radius + 1))

            if best_fit_prob >= threshold:
                return(np.array([bxx + x - radius, byy + y - radius]).astype(int))
            else:
                return(np.array([-1, -1]).astype(int))
        else:
            return(np.array([-1, -1]).astype(int))
    elif label == 2:
        for i in range(2 * radius + 1):
            for j in range(2 * radius + 1):
                left = x + i - int(part_width / 2)
                right = left + part_width
                top = y + j - int(part_width / 2)
                bottom = top + part_width
                segment = img[top:bottom, left:right]
                
                if segment.shape[0] == part_width and segment.shape[1] == part_width:
                    segment = segment.reshape((1,-1))
                    particles = np.append(particles, segment, axis=0)
                else:
                    particles = np.append(particles, np.zeros((1, part_width * part_width))-1,axis=0)
        if particles.shape[0] != 0:        
            pred_prob = svm2.predict_proba(particles)
            # here assume the correct label of pred_prob is at 1 along axis 1
            best_fit_index = np.argmax(pred_prob[:,1])
            best_fit_prob = pred_prob[best_fit_index,1]
            bxx = int(best_fit_index / (2 * radius + 1))
            byy = int(best_fit_index % (2 * radius + 1))
            # here the best fit index probability is set to -1 to find the second fit index
            pred_prob[best_fit_index,1] = -1
            second_fit_index = np.argmax(pred_prob[:,1])
            second_fit_prob = pred_prob[second_fit_index,1]
            sxx = int(second_fit_index / (2 * radius + 1))
            syy = int(second_fit_index % (2 * radius + 1))

            if second_fit_prob >= threshold:
                ret = np.array([[bxx + x - radius, byy + y - radius], [sxx + x - radius, syy + y - radius]]).astype(int)
                return ret
            elif best_fit_prob >= threshold:
                return(np.array([bxx + x - radius, byy + y - radius]).astype(int))
            else:
                return(np.array([-1, -1]).astype(int))
        else:
            return(np.array([-1, -1], [-1, -1]).astype(int))
    else:
        return(np.array([-1, -1]))
##    for i in range(2 * radius + 1):
##        for j in range(2 * radius + 1):
##            left = x + i - int(part_width / 2)
##            right = left + part_width
##            top = y + j - int(part_width / 2)
##            bottom = top + part_width
##            segment = img[top:bottom, left:right]
##            
##            if segment.shape[0] == part_width and segment.shape[1] == part_width:
##                segment = segment.reshape((1,-1))
##                particles = np.append(particles, segment, axis=0)
##            else:
##                particles = np.append(particles, np.zeros((1, part_width * part_width))-1,axis=0)
##    if particles.shape[0] != 0:        
##        pred_prob = svm.predict_proba(particles)
##        # here assume the correct label of pred_prob is at 1 along axis 1
##        best_fit_index = np.argmax(pred_prob[:,1])
##        best_fit_prob = pred_prob[best_fit_index,1]
##        bxx = int(best_fit_index / (2 * radius + 1))
##        byy = int(best_fit_index % (2 * radius + 1))
##        # here the best fit index probability is set to -1 to find the second fit index
##        pred_prob[best_fit_index,1] = -1
##        second_fit_index = np.argmax(pred_prob[:,1])
##        second_fit_prob = pred_prob[second_fit_index,1]
##        sxx = int(second_fit_index / (2 * radius + 1))
##        syy = int(second_fit_index % (2 * radius + 1))
##
##        if second_fit_prob >= threshold:
##            return(np.array([2, bxx + x - radius, byy + y - radius, sxx + x - radius, syy + y - radius]).astype(int))
##        elif best_fit_prob >= threshold:
##            return(np.array([1, bxx + x - radius, byy + y - radius, -1, -1]).astype(int))
##        else:
##            return(np.array([0, -1, -1 , -1, -1]).astype(int))
##    else:
##        return(np.array([0, -1, -1 , -1, -1]).astype(int))

'''
@ This function is used to check the similarity of two array that might
@ have different length, assume two array are 2D
@ a1      The first array
@ a2      The second array
@ repeat  Repeat time to randomly select the same size array in the longer array
@ return  Average distance between two array
'''
def check_similarity(a1, a2, threshold):
    l1 = a1.shape[0]
    l2 = a2.shape[0]

    ssize = 0
    lsize = 0
    
    if l1 > l2:
        ssize = l2
        lsize = l1
        sarray = a2.copy()
        larray = a1.copy()
    else:
        ssize = l1
        lsize = l2
        sarray = a1.copy()
        larray = a2.copy()

    count = 0

    for s in sarray:
        for l in larray:
            dist = np.linalg.norm(s - l)
        
            if dist <= threshold:
                count += 1

                break

    return count / lsize #before is ssize
    
##
##    distance = 0
##
##    sarray = sarray[sarray[:,0].argsort()]
##    larray = larray[larray[:,0].argsort()]
##
##    if ssize != lsize:
##        for r in range(repeat):
##            sample = np.random.choice(lsize, lsize - ssize, replace=False)
##            larray_copy = np.delete(larray, sample, axis = 0)
##            distance = distance + np.linalg.norm(sarray-larray_copy)
##
##        distance = distance / repeat
##    else:
##        distance = np.linalg.norm(sarray-larray)
##
##    print("sarry", sarray[0:10,:])
##    print("larry", larray[0:10,:])
##
##    return distance / ssize

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports


# Import datasets, classifiers and performance metrics
date1 = "04_06_2019"
date2 = "04_08_2019"
# The particles dataset
particles_one = np.loadtxt("particle images" + date1 + ".txt", delimiter =' ')[:2000,]
particles_two = np.loadtxt("particle images" + date2 + ".txt", delimiter =' ')

noise_one = np.loadtxt("noise" + date1 + ".txt", delimiter = ' ')[:2000,]
noise_two = np.loadtxt("noise" + date2 + ".txt", delimiter = ' ')

particles_all = np.append(particles_one, particles_two, axis=0)
noise_all = np.append(noise_one, noise_two, axis=0)

image_size_one = particles_one.shape[0]
image_size_two = particles_two.shape[0]
image_size_all = image_size_one + image_size_two

labels_one = np.ones((int(image_size_one),1))
labels_two = np.ones((int(image_size_two),1)) + 1

labels_part_all = np.append(labels_one, labels_two, axis=0)
labels_noise_all = np.zeros((image_size_all,1))

labels_two_one = np.ones((int(image_size_two),1))

labels_part_all_one = np.append(labels_one, labels_two_one, axis=0)

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
#images_and_labels = list(zip(digits.images, digits.target))
particle_and_labels_all = list(zip(particles_all, labels_part_all))
noise_and_labels_all = list(zip(noise_all, labels_noise_all))

particle_and_labels_one = list(zip(particles_one, labels_one))
particle_and_labels_two = list(zip(particles_two, labels_two))

##noise_and_labels_one = list(zip(noise_one, labels_one))
##noise_and_labels_two = list(zip(noise_one, labels_one))
images_per_row = 8

for index, (image, label) in enumerate(particle_and_labels_all[:images_per_row]):
    plt.subplot(2, images_per_row, index + 1)
    plt.axis('off')
    plt.imshow(image.reshape((20,20)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples_all = image_size_all + labels_noise_all.shape[0]

##data = particle_and_labels + noise_and_labels
row_train_index_all = int(n_samples_all * 4 / 5)

particles_data_all = np.append(particles_all, labels_part_all, axis = 1)
noise_data_all = np.append(noise_all, labels_noise_all, axis = 1)

data_all = np.append(particles_data_all, noise_data_all, axis = 0)

np.random.shuffle(data_all)
print(data_all[:,-1])
labels_all = data_all[:,-1]
data_all = np.delete(data_all, -1, axis=1)

print(data_all.shape)
print(labels_all.shape)

# Create a classifier: a support vector classifier with 0320 0324
classifier1 = svm.SVC(gamma='scale',probability = True)
# We learn the digits on the first half of the digits
##classifier.fit(data[:n_samples // 2, :], labels[:n_samples // 2])
classifier1.fit(data_all[:row_train_index_all, :], labels_all[:row_train_index_all])

data_two = np.append(particles_two, labels_two - 1, axis = 1)
data_two = np.append(data_two, noise_data_all, axis = 0)

np.random.shuffle(data_two)
labels_fit_two = data_two[:,-1]
data_two = np.delete(data_two, -1, axis=1)

n_samples_two = image_size_two + labels_noise_all.shape[0]
row_train_index_two = int(n_samples_two * 4 / 5)

# svm with 0324 only
classifier2 = svm.SVC(gamma='scale',probability = True)
classifier2.fit(data_two[:row_train_index_two, :], labels_fit_two[:row_train_index_two])

# Now predict the value of the digit on the second half:
expected = labels_all[row_train_index_all:]
predicted = classifier1.predict(data_all[row_train_index_all:,:])

print(predicted)
print("Classification report for classifier %s:\n%s\n"
      % (classifier1, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(data_all[row_train_index_all:,:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:images_per_row]):
    plt.subplot(2, images_per_row, index + images_per_row + 1)
    plt.axis('off')
    plt.imshow(image.reshape((20,20)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

# Here is the classifier with 0320 and 0324 and all particles are labeled "one"
particles_data_all_one = np.append(particles_all, labels_part_all_one, axis = 1)
data_two_one = np.append(particles_data_all_one, noise_data_all, axis = 0)

np.random.shuffle(data_two_one)
labels_fit_two_one = data_two_one[:,-1]
data_two_one = np.delete(data_two_one, -1, axis=1)

n_samples_two_one = image_size_two + labels_noise_all.shape[0]
row_train_index_two_one = int(n_samples_two_one * 4 / 5)

classifier3 = svm.SVC(gamma='scale',probability = True)
classifier3.fit(data_two_one[:row_train_index_two_one, :], labels_fit_two_one[:row_train_index_two_one])


'''
The following is the program to calculate the probability of detecting a particle
1. The images and coords are imported first
2. The SIFT points are calculated then a sliding window is created around the SIFT point
3. Then the window with largest probability is chosed as the particle centraled image
4. The central point coords are selected as the particle image centroid coords
5. The norm difference of calculated coords and real corods are calculated.
'''


single_folder = "./" + date1
double_folder = "./" + date2

single_coord_names = sorted(glob.glob(single_folder + "/image_coords/*.txt"))
single_image_names = sorted(glob.glob(single_folder + "/images/*.png"))

double_coord_names = sorted(glob.glob(double_folder + "/image_coords/*.txt"))
double_image_names = sorted(glob.glob(double_folder + "/images/*.png"))

single_size = len(single_coord_names)
double_size = len(double_coord_names)

size_all = single_size + double_size

coord_names_all = single_coord_names + double_coord_names
image_names_all = single_image_names + double_image_names

single_image_coords_calculated = []
double_image_coords_calculated = []

for i in range(double_size):
    img = cv2.imread(double_image_names[i],0)
    img = gaussian_filter(img, 0.5)
    thresh = cv2.threshold(img, 50, 255, 0)[1]

    coords = np.empty((0,2))
    coords_svm = np.empty((0,2))
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    size = 0
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            size += 1
            coords = np.append(coords, np.array([cX,cY]).reshape((1,2)), axis=0)

            ret = move_particle(img, np.array([cX, cY]),
                                classifier1, classifier2, classifier3, 10, 20, 0.95).reshape((-1,2))
            if (ret > 0).all() :
##                print("ret is: ", ret)
                coords_svm = np.append(coords_svm, ret, axis=0)            
    single_image_coords_calculated.append(coords)
    read_coords = np.loadtxt(double_coord_names[i],delimiter=' ')
    print("There are %d countours", {size})
    print("There are coords svm ", coords_svm.shape[0])
    print("distance between real coords and coords svm is: ",
          check_similarity(read_coords, coords_svm, 1))
    print("distance between real coords and coords is: ",
          check_similarity(read_coords, coords, 1))
##    
##for i in range(size_all):
##    img = cv2.imread(image_names_all[i],0)
##    img = gaussian_filter(img, 0.5)
##    thresh = cv2.threshold(img, 100, 255, 0)[1]
##
##    coords = np.empty((0,2))
##    coords_svm = np.empty((0,2))
##    # find contours in the thresholded image
##    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
##    cnts = imutils.grab_contours(cnts)
##    size = 0
##    for c in cnts:
##        # compute the center of the contour
##        M = cv2.moments(c)
##        if M["m00"] != 0:
##            cX = int(M["m10"] / M["m00"])
##            cY = int(M["m01"] / M["m00"])
##            size += 1
##            coords = np.append(coords, np.array([cX,cY]).reshape((1,2)), axis=0)
##
##            ret = move_particle(img, np.array([cX, cY]),
##                                classifier1, classifier2, classifier3, 10, 20, 0.99).reshape((-1,2))
##            if (ret > 0).all() :
####                print("ret is: ", ret)
##                coords_svm = np.append(coords_svm, ret, axis=0)            
##    single_image_coords_calculated.append(coords)
##    read_coords = np.loadtxt(single_coord_names[i],delimiter=' ')
##    print("There are %d countours", {size})
##    print("There are coords svm ", coords_svm.shape[0])
##
##    print("distance between real coords and coords svm is: ",
##          check_similarity(read_coords, coords_svm, 1))
##    print("distance between real coords and coords is: ",
##          check_similarity(read_coords, coords, 1))
    
##    if i == 0:
##        print("---",coords_svm[coords_svm[:,0].argsort()])
##        print("---",read_coords[read_coords[:,0].argsort()])
##        break

    
cv2.destroyAllWindows()

