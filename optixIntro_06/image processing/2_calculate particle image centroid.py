import glob
import numpy as np
import cv2
from projection import projection     

folder = "../data/back up data/05_30_2019"

coord_names = sorted(glob.glob(folder + "/coordinates/*.txt"))
image_names = sorted(glob.glob(folder + "/images/*.png"))

##coord_names = sorted(glob.glob("../data/back up data/03_20_2019/coordinates/*.txt"))
##image_names = sorted(glob.glob("../data/back up data/03_20_2019/images/*.png"))

size = len(coord_names)
print(size)
zeros = len(str(size))

for i in range(size):
    image = cv2.imread(image_names[i],0)
    image_shape = image.shape

    data = np.loadtxt(coord_names[i],delimiter = ' ')
    coords = data[:,0:3]
    radii = data[:,-1]
    part_num = data.shape[0]

    image_coords = np.zeros((part_num,2))

    for j in range(part_num):
        coord = coords[j,]
        radius = radii[j]
        temp_coord = projection((coord))
        temp_coord[1] = 1023 - temp_coord[1]
        image_coords[j,:] = temp_coord

    image_coords_name = folder + "/image_coords/" + '0'*(zeros-len(str(i)))  + str(i) + '.txt'


##    image_coords_name = "../data/back up data/03_20_2019/image_coords/" + '0'*(zeros-len(str(i)))  + str(i) + '.txt'
        
    with open(image_coords_name, 'wb') as f:
        np.savetxt(f,image_coords,fmt="%d",delimiter = ' ')
    
