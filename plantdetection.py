from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv

I = imread('young-corn-plants-in-a-rows.jpeg')

# J = rgb_to_hsv(I)
#
# L = [I, J]
# for K in L:
#     size = K.shape
#     for i in range(size[2]):
#         plt.figure()
#         plt.axis("off")
#         plt.imshow(K[:,:,i], interpolation='bicubic', cmap='gray')

#
I = I/255
R = I[:,:,0]
G = I[:,:,1]
B = I[:,:,2]

ExG = 2*G - R - B
T_ExG = ExG<0.3

CIVE = 0.441*R - 0.881*G + 0.385*B + 18.78745/255
T_CIVE = CIVE>-0.1

L = [I, ExG, T_ExG, CIVE, T_CIVE]
Labels = ['original', 'Excess green',
          'Limiarized excess green',
          'Color index of vegetation (CIVE)',
          'Limiarized CIVE']
for i, K in enumerate(L):
    plt.figure()
    plt.axis("off")
    plt.suptitle(Labels[i])
    plt.imshow(K, interpolation='bicubic', cmap='gray')

plt.pause(99999)




