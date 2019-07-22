"""
================================
Recognizing hand-written digits
================================

An example showing how the scikit-learn can be used to recognize images of
hand-written digits.

This example is commented in the
:ref:`tutorial section of the user manual <introduction>`.

"""
print(__doc__)



# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The particles dataset
particles = np.loadtxt("particle images_0320.txt", delimiter =' ')
particles_new = np.loadtxt("particle images_0324.txt", delimiter =' ')

noise = np.loadtxt("noise particles_0320.txt", delimiter = ' ')
noise_new = np.loadtxt("noise particles_0324.txt", delimiter = ' ')

particles = np.append(particles, particles_new, axis=0)
noise = np.append(noise, noise_new, axis=0)

image_size = particles.shape[0]

labels_one = np.ones((int(image_size/2),1))
labels_two = labels_one + 1

labels_part = np.append(labels_one, labels_two, axis=0)
labels_noise = np.zeros((image_size,1))


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

row_train_index = int(n_samples * 4 / 5)

# We learn the digits on the first half of the digits
##classifier.fit(data[:n_samples // 2, :], labels[:n_samples // 2])
classifier.fit(data[:row_train_index, :], labels[:row_train_index])

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

def move_particle(particle_array):
    particle_array = particle_array.reshape((1,-1))
    width = particle_array.shape[1]
    ret = np.empty((0,width))
##    print("ret shape is: ",ret.shape)
##    print("particle array shape is: ",particle_array.shape)
    for part in particle_array:
        particle = part.copy()
##        print(particle.shape)
        width = np.sqrt(len(particle)).astype(int)
        particle = particle.reshape((width, width))
##        print(particle.shape)

        particle_padded = np.pad(particle,pad_width = ((width, width),(width, width)),mode='median')

        for i in range(10, (width + 1)):
            for j in range(10, (width + 1)):
                particle = particle_padded[i:i+20,j:j+20]
                ret = np.append(ret,particle.reshape((1,-1)),axis=0)

    return ret

        
        

##moved_particles = move_particle(particles[0,:])

import cv2

##for index, part in enumerate(moved_particles):
##    if index <= 1000:
##        part = part.reshape((20,20))/255
##        cv2.imshow("part"+str(index),part)
##        pred_prob = classifier.predict_proba(moved_particles[index].reshape((1,-1)))
##        print(str(index),pred_prob)
##        cv2.waitKey(0)
##cv2.destroyAllWindows()

##print("moved particles shape ",moved_particles.shape)
##print("moved_particles",moved_particles[0,:])
##pred_prob = classifier.predict_proba(moved_particles)
##print(pred_prob)

plt.show()
