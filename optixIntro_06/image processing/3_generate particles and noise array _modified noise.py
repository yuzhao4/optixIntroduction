import glob
import numpy as np
import cv2
from projection import projection     

folder = "../data/back up data/"
date = "05_30_2019"

coord_names = sorted(glob.glob(folder + date + "/coordinates/*.txt"))
image_names = sorted(glob.glob(folder + date + "/images/*.png"))

imgcoord_names = sorted(glob.glob(folder + date + "/image_coords/*.txt"))
##image_names = sorted(glob.glob("../data/back up data/03_24_2019/images/*.png"))

print(imgcoord_names)

size = len(imgcoord_names)
rad = 10

def segment(img, x, y,radius):
    dim = img.shape
    #print("img dim is: ",dim)

    if x + radius < dim[1] and y + radius < dim[0] and x - radius > 0 and y - radius > 0:
        left  =  (x - radius).astype(int)
        right =  (x + radius).astype(int)
        up    =  (y - radius).astype(int)
        down  =  (y + radius).astype(int)
        
        particle = img[up:down,left:right]
        #print("particle shape is: ",particle.shape)
        return particle
    else:
        print("particle image out of bound")
        return(np.zeros((rad*2,rad*2))-1)

def segment_noise(img, x, y,radius):
    dim = img.shape
    #print("img dim is: ",dim)
    movement = np.random.uniform(-radius, radius, 2)

    x = x + movement[0]
    y = y + movement[1]
    
    if x + radius < dim[1] and y + radius < dim[0] and x - radius > 0 and y - radius > 0:
        left  =  (x - radius).astype(int)
        right =  (x + radius).astype(int)
        up    =  (y - radius).astype(int)
        down  =  (y + radius).astype(int)
        
        particle = img[up:down,left:right]
        #print("particle shape is: ",particle.shape)
        return particle
    else:
        print("particle image out of bound")
        return(np.zeros((rad*2,rad*2))-1)

def generate_noise(num, rad, left, right):
    if num == 0:
        ret = np.zeros((0,(rad*2)**2))
    else:
        ret = np.random.uniform(left, right, (num,(rad*2)**2))
    return ret

##noise = generate_noise(0,rad,0,0.2)*255

modified_noise = np.zeros((0, (rad*2)**2))
single_particles = np.zeros((0,(rad*2)**2))
##for image in noise:
##    cv2.imshow("noise", image)
##    cv2.waitKey(0)
##
##cv2.destroyAllWindows()

for i in range(size):
    image = cv2.imread(image_names[i],0)
    image_shape = image.shape
    print("i is: ", i)
    coords = np.loadtxt(imgcoord_names[i],delimiter = ' ')
    #print("coords is ",coords)
    for coord in coords:
##        print("coords is: ",coord)
        particle_pic = segment(image, coord[0],coord[1],rad).reshape((1,-1))
##        single_particles[i,:] = particle_pic
        single_particles = np.append(single_particles, particle_pic, axis=0)
##        noise = np.append(noise, generate_noise(1,rad,0,0.2)*255,axis=0)
        particle_noise = segment_noise(image, coord[0],coord[1],rad)
##        cv2.imshow("wtf", particle_noise)
##        cv2.waitKey(0)
##        cv2.destroyAllWindows()
        particle_noise = particle_noise.reshape((1,-1))
        modified_noise = np.append(modified_noise, particle_noise, axis = 0)
##        cv2.imshow("what",particle_pic)
##        cv2.waitKey(0)

def remove_neg(array):
    index = np.where(array[:, 0] < 0)
    ret = np.delete(array, index, axis=0)
    return ret

single_particles = remove_neg(single_particles)
modified_noise = remove_neg(modified_noise)

##np.savetxt("noise particles_0406.txt", noise, fmt="%d", delimiter=' ')
np.savetxt("noise" + date + ".txt", modified_noise, fmt="%d", delimiter=' ')
np.savetxt("particle images" + date + ".txt", single_particles, fmt="%d", delimiter=' ')

print(np.loadtxt("particle images" + date + ".txt").shape)
