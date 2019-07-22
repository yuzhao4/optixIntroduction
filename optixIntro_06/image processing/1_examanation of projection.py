import glob
import numpy as np
import cv2
from projection import projection     

folder = "../data/back up data/05_30_2019"

coord_names = sorted(glob.glob(folder + "/coordinates/*.txt"))
image_names = sorted(glob.glob(folder + "/images/*.png"))

##coord_names = sorted(glob.glob("../data/back up data/03_24_2019/coordinates/*.txt"))
##image_names = sorted(glob.glob("../data/back up data/03_24_2019/images/*.png"))

size = len(coord_names)


for i in range(size):
    image = cv2.imread(image_names[i],0)
    image_shape = image.shape

    data = np.loadtxt(coord_names[i],delimiter = ' ')
    coords = data[:,0:3]
    radii = data[:,-1]
    print("********************************")
    for coord,radius in zip(coords,radii):
        #coord = np.array([0,0,0]).astype(int)
        
        #print("coords is", coord)
        image_coord = projection((coord))
##        print("projected coord is ", image_coord)
##        cv2.circle(image, image_coord, 10, 255, -1)
        
##        cv2.circle(image, (image_coord[0],image_coord[1]), int(radius*2), (255,0,0), 1)
##        cv2.circle(image, (image_shape[1] - 1 - image_coord[0],image_coord[1]),
##                   int(radius*2), (0,0,255), 1)
        cv2.circle(image, (image_coord[0],1023-image_coord[1]),
                   int(radius*6), (255,0,0), 1)
##        
##        cv2.circle(image, (639,539), int(radius*8), (255,0,0), 1)
##        cv2.circle(image, (10,20), int(radius*8), (255,0,0), 1)
##        cv2.circle(image, (20,40), int(radius*8), (255,0,0), 1)
##        cv2.circle(image, (30,60), int(radius*8), (255,0,0), 1)
##        cv2.circle(image, (40,80), int(radius*8), (255,0,0), 1)
##        cv2.circle(image, (1279,1023), int(radius*8), (255,0,0), 1)


    cv2.namedWindow(str(i), cv2.WINDOW_AUTOSIZE)
    cv2.imshow(image_names[i], image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
    
