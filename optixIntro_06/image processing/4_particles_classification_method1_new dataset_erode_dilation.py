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
from joblib import dump, load
import time

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
@ img         The img with particles with shape((x,x))
@ centroid    The calculated centroid passed to shake
@ svm         The svm model used to calculate the probability
@ radius      The radius to move around the centroid
@ part_width  The particle image width
@ threshold   The threshold to determine to return one or two probability 
@ return      The moved single particle image with shape((1, -1))
'''
def move_particle(image, centroid, svm, radius, part_width, threshold):
    img = image.copy()
    img = np.pad(img, pad_width=((radius,radius),(radius,radius)),mode='median')
    particles = np.empty((0, part_width * part_width))
    # can I replace following loop to increase the speed??

    x = centroid[0]
    y = centroid[1]

    # x is width, y is height
##    print("x is: ", x)
##    print("y is: ", y)
    if radius == 0:
        print("radius is zero")
        exit()
    for i in range(2 * radius + 1):
        for j in range(2 * radius + 1):
            left = x + i - int(part_width / 2)
            right = left + part_width
            top = y + j - int(part_width / 2)
            bottom = top + part_width
##            segment = img[left:right, top:bottom]
            segment = img[top:bottom, left:right]
            
##            print("img shape: ", img.shape)
            if segment.shape[0] == part_width and segment.shape[1] == part_width:
##                print("segment shape is: ", segment.shape)
                segment = segment.reshape((1,-1))
                particles = np.append(particles, segment, axis=0)
            else:
                particles = np.append(particles, np.zeros((1, part_width * part_width))-1,axis=0)
    if particles.shape[0] != 0:        
        pred_prob = svm.predict_proba(particles)
        # here assume the correct label of pred_prob is at 1 along axis 1
        best_fit_index = np.argmax(pred_prob[:,1])
        best_fit_prob = pred_prob[best_fit_index,1]
##        print("best prob is: ", best_fit_prob)
        bxx = int(best_fit_index / (2 * radius + 1))
        byy = int(best_fit_index % (2 * radius + 1))
##        print("best x is %d, y is %d" % (bxx,byy)) 

        # here the best fit index probability is set to -1 to find the second fit index
        pred_prob[best_fit_index,1] = -1
        second_fit_index = np.argmax(pred_prob[:,1])
        second_fit_prob = pred_prob[second_fit_index,1]
##        print("second prob is: ", second_fit_prob)
        sxx = int(second_fit_index / (2 * radius + 1))
        syy = int(second_fit_index % (2 * radius + 1))
##        print("second x is %d, y is %d" % (sxx,syy))

        if second_fit_prob >= threshold:
            return(np.array([2, bxx + x - radius, byy + y - radius, sxx + x - radius, syy + y - radius]).astype(int))
        elif best_fit_prob >= threshold:
            return(np.array([1, bxx + x - radius, byy + y - radius, -1, -1]).astype(int))
        else:
            return(np.array([0, -1, -1 , -1, -1]).astype(int))
    else:
        return(np.array([0, -1, -1 , -1, -1]).astype(int))

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
        
            if dist < threshold:
                count += 1

                break

    return count / lsize
    
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

# The particles dataset
date1 = "2000"
##date2 = "04_08_2019"

particles = np.loadtxt("particle images" + date1 + ".txt", delimiter =' ')
##print(particles.shape)
##particles_new = np.loadtxt("particle images" + date2 + ".txt", delimiter =' ')

noise = np.loadtxt("noise" + date1 + ".txt", delimiter = ' ')
##noise_new = np.loadtxt("noise" + date2 + ".txt", delimiter = ' ')

##particles = np.append(particles, particles_new, axis=0)
##noise = np.append(noise, noise_new, axis=0)

image_size = particles.shape[0]
noise_size = noise.shape[0]

##labels_one = np.ones((int(image_size/2),1))
##labels_two = labels_one

##labels_part = np.append(labels_one, labels_two, axis=0)
labels_part = np.ones((int(image_size),1))
labels_noise = np.zeros((int(noise_size),1))

image_size = image_size + noise_size

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
#images_and_labels = list(zip(digits.images, digits.target))
particle_and_labels = list(zip(particles, labels_part))
noise_and_labels = list(zip(noise, labels_noise))

images_per_row = 8
print("line 232")
for index, (image, label) in enumerate(particle_and_labels[:images_per_row]):
    plt.subplot(2, images_per_row, index + 1)
    plt.axis('off')
    plt.imshow(image.reshape((20,20)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = image_size
##data = particle_and_labels + noise_and_labels

particles_data = np.append(particles, labels_part, axis = 1)
noise_data = np.append(noise, labels_noise, axis = 1)

data = np.append(particles_data, noise_data, axis = 0)

np.random.shuffle(data)
print(data[:,-1])
labels = data[:,-1]
data = np.delete(data, -1, axis=1)

print(data.shape)
print(labels.shape)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma='scale',probability = True)
##classifier = load('04_08 classifier')
row_train_index = int(n_samples * 4 / 5)

# We learn the digits on the first half of the digits
##classifier.fit(data[:n_samples // 2, :], labels[:n_samples // 2])
classifier.fit(data[:row_train_index, :], labels[:row_train_index])
dump(classifier, '04_08_single & double classifier')
# Now predict the value of the digit on the second half:
expected = labels[row_train_index:]
predicted = classifier.predict(data[row_train_index:,:])

print(predicted)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(data[row_train_index:,:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:images_per_row]):
    plt.subplot(2, images_per_row, index + images_per_row + 1)
    plt.axis('off')
    plt.imshow(image.reshape((20,20)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

'''
The following is the program to calculate the probability of detecting a particle
1. The images and coords are imported first
2. The SIFT points are calculated then a sliding window is created around the SIFT point
3. Then the window with largest probability is chosed as the particle centraled image
4. The central point coords are selected as the particle image centroid coords
5. The norm difference of calculated coords and real corods are calculated.
'''


single_folder = "./" + date1
##double_folder = "./" + date2

single_coord_names = sorted(glob.glob(single_folder + "/image_coords/*.txt"))
single_image_names = sorted(glob.glob(single_folder + "/images/*.png"))

##double_coord_names = sorted(glob.glob(double_folder + "/image_coords/*.txt"))
##double_image_names = sorted(glob.glob(double_folder + "/images/*.png"))

single_size = len(single_coord_names)
##double_size = len(double_coord_names)

single_image_coords_calculated = []
##double_image_coords_calculated = []

##for i in range(double_size):
##    img = cv2.imread(double_image_names[i],0)
####    cv2.imshow(str(i),img)
####    cv2.waitKey(0)
##    img = gaussian_filter(img, 0.5)
##    thresh = cv2.threshold(img, 50, 255, 0)[1]
##
##    coords = np.empty((0,2))
##    coords_svm = np.empty((0,5))
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
##                                classifier, 10, 20, 0.95).reshape((1,5))
##            coords_svm = np.append(coords_svm, ret, axis=0)            
##    single_image_coords_calculated.append(coords)
##    read_coords = np.loadtxt(double_coord_names[i],delimiter=' ')
##    print("There are %d countours", {size})
##    print("distance between real coords and coords svm I is: ",
##          check_similarity(read_coords, coords_svm[:,1:3], 1))
##    print("distance between real coords and coords svm II is: ",
##          check_similarity(read_coords, coords_svm[:,3:5], 1))
##    print("distance between real coords and coords is: ",
##          check_similarity(read_coords, coords, 1))
    
##    if i <=3:
##        print("---",coords_svm[:,1:3][coords_svm[:,1].argsort()])
##        print("---",coords_svm[:,3:5][coords_svm[:,3].argsort()])
##        print("---",read_coords[read_coords[:,0].argsort()])
##    else:
##        break

svm_accuracy = []
cnt_accuracy = []
all_time = []
num_svm = []
num_cnt = []

radius = 5

time1 = time.time()
all_time.append(time1)
print("single size ", single_size)
for i in range(single_size):
    img = cv2.imread(single_image_names[i],0)
    img = gaussian_filter(img, 0.5)
    thresh = cv2.threshold(img, 150, 255, 0)[1]
##
##    erode_kernel = np.ones((2,2))
##    dilation_kernel = np.ones((5,5))
##
##    thresh_processed = thresh.copy()
##
##    thresh_processed = cv2.dilate(thresh_processed, dilation_kernel, iterations=1)    
##    thresh = cv2.erode(thresh_processed, erode_kernel, iterations=2)

##    cv2.imshow("thresh", thresh)
##    cv2.imshow("img", img)
##    cv2.imshow("thresh_processed", thresh_processed)

    
    coords = np.empty((0,2))
    coords_svm = np.empty((0,2))
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    img_cnt = img.copy()
##    cv2.drawContours(img_cnt, cnts, 1, (255,0,0), 1)
##    cv2.imshow("img_cnt", img_cnt)
##    cv2.imshow("img", img)
##
##    cv2.waitKey(0)
##    break
    

    size = 0
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            size += 1
            coords = np.append(coords, np.array([cX,cY]).reshape((1,2)), axis=0)
##            cv2.circle(img_cnt, (cX, cY),
##                       int(10), (255,0,0), 1)
            ret = move_particle(img, np.array([cX, cY]),
                                classifier, radius, 20, 0.95).reshape((1,5))
            if ret[0,0] == 2 and np.linalg.norm(ret[1:3] - ret[3:5]) <= 1:
                ret[0,3] = -1
                ret[0,4] = -1
##            print("ret is: ", ret)            
            ret = ret[:,1:5].reshape((2,2))
##            print("reshape ret is: ", ret)            

            coords_svm = np.append(coords_svm, ret, axis=0)

##    cv2.imshow("cnts", img_cnt)
##    cv2.imshow("img", img)
 
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

    all_time.append(time.time()-all_time[0])

    single_image_coords_calculated.append(coords)
    read_coords = np.loadtxt(single_coord_names[i],delimiter=' ')
    print("There are %d countours", {size})
    num_cnt.append(size)
    index = np.where(coords_svm[:,0] == -1)
    coords_svm = np.delete(coords_svm, index, axis = 0)
    print("coords svm is: ", coords_svm)
    
    print("There are svm coords", coords_svm.shape[0])
    num_svm.append(coords_svm.shape[0])
    print("distance between real coords and coords svm is: ",
          check_similarity(read_coords, coords_svm, 1))
    svm_accuracy.append(check_similarity(read_coords, coords_svm, 1))
    print("distance between real coords and coords is: ",
          check_similarity(read_coords, coords, 1))
    cnt_accuracy.append(check_similarity(read_coords, coords, 1))
    
##    if i == 5:
##        break
time2 = time.time()

##operation_time = (time2 - time1)/5
##print("operation time is: ", operation_time)
all_time = all_time[1:]

time_len = len(all_time)

for i in range(time_len - 1,0,-1):
    all_time[i] = all_time[i] - all_time[i-1]

print("svm_accuracy is: ", svm_accuracy)
print("cnt_accuracy is: ", cnt_accuracy)
print("time is: ", all_time)
print("num_svm is: ", num_svm)
print("num_cnt is: ", num_cnt)

svm_accuracy = np.array(svm_accuracy).reshape((1,single_size))
cnt_accuracy = np.array(cnt_accuracy).reshape((1,single_size))
all_time = np.array(all_time).reshape((1,single_size))
num_svm = np.array(num_svm).reshape((1,single_size))
num_cnt = np.array(num_cnt).reshape((1,single_size))

result = np.zeros((0,single_size))
result = np.append(result, svm_accuracy, axis=0)
result = np.append(result, cnt_accuracy, axis=0)
result = np.append(result, all_time, axis=0)
result = np.append(result, num_svm, axis=0)
result = np.append(result, num_cnt, axis=0)

print("result is: ", result)

mean = np.mean(result, axis = 1).reshape((1,5))
std = np.std(result, axis = 1).reshape((1,5))

ret = np.append(mean, std, axis = 0)

head = "svm_accuracy\ncnt_accuracy\ntime\nnum_svm\nnum_cnt"

np.savetxt("result "+ date1 + " radius " + str(radius) + ".txt", ret, delimiter = ' ', header = head)

cv2.destroyAllWindows()

