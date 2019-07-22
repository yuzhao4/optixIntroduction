import numpy as np
import subprocess

W = 150
H = 150
D = 100
PART_NUM = 500

def generate_particle(W,H,D,PART_NUM):
    particles_coords = np.zeros((PART_NUM,4))

    particles_coords[:,0] = np.random.uniform(-W, W, PART_NUM)
    particles_coords[:,1] = np.random.uniform(-H, H, PART_NUM)
    particles_coords[:,2] = np.random.uniform(-D, D, PART_NUM)

    particles_coords[:,-1] = np.random.normal(1,0.1,PART_NUM)

    return particles_coords

IMAGE_NUM = 8

#zerosNumMax = PART_NUM / 10

count = 0
##zerosNumMax = len(str(abs(IMAGE_NUM * 1000)))
zerosNumMax = len(str(abs(1000)))

for i in range(IMAGE_NUM):
    particles = generate_particle(W,H,D,PART_NUM)

    zeros = '0' * int(zerosNumMax - len(str(i)) - 1) 
    
    
    txtname = "C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\build 64\\output_data\\coordinates\\" + "particles_generated" + zeros + str(i) + ".txt"   

    pngname = "C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\build 64\\output_data\\images\\" + "particles_generated" + zeros + str(i) + ".png"
    
    np.savetxt(txtname, particles,fmt='%.3f',
               delimiter=' ',newline='\n')

    np.savetxt("C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\src\\optixIntroduction\\optixIntro_06\\data\\particles_generated.txt", particles,fmt='%.3f',
               delimiter=' ',newline='\n')


    subprocess.run(["C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\build 64\\bin\\Release\\optixIntro_06", "-l","-f",pngname])
