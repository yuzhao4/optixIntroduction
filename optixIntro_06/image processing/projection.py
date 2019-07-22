import numpy as np

RT1 = np.loadtxt("RT1.txt", delimiter=',')
RT2 = np.loadtxt("RT2.txt", delimiter=',')
RT3 = np.loadtxt("RT3.txt", delimiter=',')
RT4 = np.loadtxt("RT4.txt", delimiter=',')

R1 = RT1[0:3,:]
T1 = RT1[3,:]
R2 = RT2[0:3,:]
T2 = RT2[3,:]
R3 = RT3[0:3,:]
T3 = RT3[3,:]
R4 = RT4[0:3,:]
T4 = RT4[3,:]

R = np.loadtxt("Xtox.txt")
T = np.loadtxt("transform.txt")

##def projection(point_coords):
##    if len(point_coords) != 3:
##        print("3 dimension need to be cast to homogeneous coord")
##    elif len(point_coords) == 3:
####        point_coords = np.dot(point_coords + T, R.T)
##        point_coords = np.append(point_coords,1.0)
##
##    fox = 12.56
##    foy = 12.56
####    K = np.array(([fox / (4.8 * 0.001),0,-640],[0,foy/ (4.8 * 0.001),-512],[0,0,-1]))
##    K = np.array(([fox / (4.8 * 0.001),0,0],[0,foy/ (4.8 * 0.001),0],[0,0,1]))
##
##    RRC = np.loadtxt("Xtox.txt",delimiter=' ')
##    transform = np.array([-220.823, 282.93, -679.623])
##    transform= transform.reshape((3,1))
##    RRC = np.append(RRC,-transform,axis=1)
##    
##    P = np.dot(K, RRC)
##
##    result = np.dot(P, point_coords.T)
##    result = result / result[-1]
##    result[1] = result[1] * -1
##    
##    result = result + np.array(([640,512,0]))
##    negative = np.where(result < 0)
##    positive = np.where(result > 0)
##
##    
##    result[negative] = np.ceil(result[negative])
##    result[positive] = np.floor(result[positive])
####    result = (np.floor(result * 1.0) ).astype(int)
####    print(result)
##    return((result[0:2]).astype(int))

def projection(point_coords):
##    trans = np.dot(R, T)
##
##    point_coords = np.dot(R, point_coords)
##    point_coords = point_coords - trans
    point_coords = np.append(point_coords,1.0)

##    print("point_coords is: ", point_coords)
##    fox = 5
##    foy = 5
##    K = np.array(([fox / (4.8 * 0.001),0,640],[0,foy/ (4.8 * 0.001),512],[0,0,1]))
    fox = 0.96
    foy = 0.96
    K = np.array(([fox/0.000368,0,640],[0,foy/0.000368,512],[0,0,1]))
    RRC = R.T
##    print("RRC IS: ", RRC)
    transform = -np.array([0,0,-7.68570038e+02])
    transform= transform.reshape((3,1))
    RRC = np.append(RRC, transform,axis=1)
    P = np.dot(K, RRC)
    
    result = np.dot(P, point_coords.T)

    result = result / result[-1]
##    print("points in local system is: ", result)
##    P = np.dot(K, RRC)
##    result[1] = result[1] * -1
##    print("result is: ", result)    
##    result = result - np.array(([640,512,0]))    
    return ((result[0:2]).astype(int))

if __name__ == '__main__':
    test_points = np.loadtxt("C:\\optix_advanced_samples-master\\optix_advanced_samples-master\\src\\optixIntroduction\\optixIntro_06\data\\particles_generated.txt",
                             delimiter = ' ')
    print()
    print(projection( np.array([-220.823, 282.93, -679.623])))
    print(projection( np.array([0, 0, 0])))
    print()
    for i in test_points[:,0:3]:
        print(projection(i))
