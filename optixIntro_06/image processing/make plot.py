import numpy as np
import matplotlib.pyplot as plt

particle_density = np.array([200,1000,2000])

reference_x = particle_density
reference_y = np.ones(reference_x.shape[0])*100

svm_accu_fls = np.array([8.025e-01,7.002500000000000391e-01,5.794999999999999041e-01])*100
cnt_accu_fls= np.array([2.4858e-01,2.137499999999999956e-01,1.839999999999999969e-01])*100

accu_diff_fls = np.mean(svm_accu_fls - cnt_accu_fls)
print("accu_diff_fls is ",accu_diff_fls)

num_svm_fls = np.array([1.815e+02,7.997500000000000000e+02,1.346500000000000000e+03])/particle_density*100
num_cnt_fls = np.array([1.997e+02,9.310000000000000000e+02,1.682000000000000000e+03])/particle_density*100

upper_ylim = 130

plt.plot(particle_density,svm_accu_fls,c='r',marker = 'o',label='svm')
plt.plot(particle_density,cnt_accu_fls,c='g',marker = 'x',label='contour')
plt.xlabel('particle number / image')
plt.ylabel('Particle Identification Accuracy (%)')
plt.title('Accuracy comparison (four lights on)')
plt.xticks([200,1000,2000])
plt.plot(reference_x, reference_y, '--',marker='', label='100%')
plt.legend()
plt.ylim([0,upper_ylim])

plt.show()

plt.plot(particle_density,num_svm_fls,c='r',marker = 'o',label='svm')
plt.plot(particle_density,num_cnt_fls,c='g',marker = 'x',label='contour')
plt.xlabel('particle number / image')
plt.ylabel('Particle Recovery Accuracy (%)')
plt.title('Particle Recovery Rate (four lights on)')
plt.xticks([200,1000,2000])
plt.plot(reference_x, reference_y, '--',marker='', label='100%')
plt.legend()
plt.ylim([0,upper_ylim])
plt.show()
#------------------------------------

svm_accu_tls = np.array([8.072500000000000231e-01,7.627500000000000391e-01,6.432499999999999885e-01])*100
cnt_accu_tls= np.array([2.940149061720555013e-01,2.562499999999999778e-01,2.277500000000000080e-01])*100

accu_diff_tls = np.mean(svm_accu_tls - cnt_accu_tls)
print("accu_diff_tls is ",accu_diff_tls)

num_svm_tls = np.array([1.798000000000000114e+02,8.637500000000000000e+02,1.496000000000000000e+03])/particle_density*100
num_cnt_tls = np.array([2.010500000000000114e+02,9.470000000000000000e+02,1.728500000000000000e+03])/particle_density*100

plt.plot(particle_density,svm_accu_tls,c='r',marker = 'o',label='svm')
plt.plot(particle_density,cnt_accu_tls,c='g',marker = 'x',label='contour')
plt.xlabel('particle number / image')
plt.ylabel('Particle Identification Accuracy (%)')
plt.title('Accuracy comparison (three lights on)')
plt.xticks([200,1000,2000])
plt.plot(reference_x, reference_y, '--',marker='', label='100%')
plt.legend()
plt.ylim([0,upper_ylim])
plt.show()

plt.plot(particle_density,num_svm_tls,c='r',marker = 'o',label='svm')
plt.plot(particle_density,num_cnt_tls,c='g',marker = 'x',label='contour')
plt.xlabel('particle number / image')
plt.ylabel('Particle Recovery Accuracy (%)')
plt.title('Particle Recovery Rate (three lights on)')
plt.xticks([200,1000,2000])
plt.plot(reference_x, reference_y, '--',marker='', label='100%')
plt.legend()
plt.ylim([0,upper_ylim])
plt.show()

#------------------------------------
svm_accu_twls = np.array([4.930000000000000493e-01,5.155000000000000693e-01,4.782499999999999529e-01])*100
cnt_accu_twls= np.array([9.347165449357910938e-02,1.084911085646379642e-01,1.206613203056945138e-01])*100

accu_diff_twls = np.mean(svm_accu_twls - cnt_accu_twls)
print("accu_diff_twls is ",accu_diff_twls)

num_svm_twls = np.array([1.228499999999999943e+02,7.082500000000000000e+02,1.196500000000000000e+03])/particle_density*100
num_cnt_twls = np.array([2.520500000000000114e+02,1.191500000000000000e+03,2.312000000000000000e+03])/particle_density*100

plt.plot(particle_density,svm_accu_twls,c='r',marker = 'o',label='svm')
plt.plot(particle_density,cnt_accu_twls,c='g',marker = 'x',label='contour')
plt.xlabel('particle number / image')
plt.ylabel('Particle Identification Accuracy (%)')
plt.title('Accuracy comparison (two lights on)')
plt.xticks([200,1000,2000])
plt.plot(reference_x, reference_y, '--',marker='', label='100%')
plt.legend()
plt.ylim([0,upper_ylim])
plt.show()

plt.plot(particle_density,num_svm_twls,c='r',marker = 'o',label='svm')
plt.plot(particle_density,num_cnt_twls,c='g',marker = 'x',label='contour')
plt.xlabel('particle number / image')
plt.ylabel('Particle Recovery Accuracy (%)')
plt.title('Particle Recovery Rate (two lights on)')
plt.xticks([200,1000,2000])
plt.plot(reference_x, reference_y, '--',marker='', label='100%')
plt.legend()
plt.ylim([0,upper_ylim])
plt.show()

#------------------------------------
svm_accu_onels = np.array([4.132499999999999507e-01,4.049999999999999711e-01,4.104999999999999760e-01])*100
cnt_accu_onels= np.array([9.699978687127025490e-02,9.575000000000000178e-02,9.024999999999999689e-02])*100

accu_diff_onels = np.mean(svm_accu_onels - cnt_accu_onels)
print("accu_diff_onels is ",accu_diff_onels)

num_svm_onels = np.array([1.139000000000000057e+02,5.830000000000000000e+02,1.182500000000000000e+03])/particle_density*100
num_cnt_onels = np.array([1.841999999999999886e+02,9.070000000000000000e+02,1.793000000000000000e+03])/particle_density*100

plt.plot(particle_density,svm_accu_onels,c='r',marker = 'o',label='svm')
plt.plot(particle_density,cnt_accu_onels,c='g',marker = 'x',label='contour')
plt.xlabel('particle number / image')
plt.ylabel('Particle Identification Accuracy (%)')
plt.title('Accuracy comparison (one light on)')
plt.xticks([200,1000,2000])
plt.plot(reference_x, reference_y, '--',marker='', label='100%')
plt.legend()
plt.ylim([0,upper_ylim])
plt.show()

plt.plot(particle_density,num_svm_onels,c='r',marker = 'o',label='svm')
plt.plot(particle_density,num_cnt_onels,c='g',marker = 'x',label='contour')
plt.xlabel('particle number / image')
plt.ylabel('Particle Recovery Accuracy (%)')
plt.title('Particle Recovery Rate (one light on)')
plt.xticks([200,1000,2000])
plt.plot(reference_x, reference_y, '--',marker='', label='100%')
plt.legend()
plt.ylim([0,upper_ylim])
plt.show()
