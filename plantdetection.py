from scipy.misc import imread, imsave
import os, glob
import numpy as np
from skimage import filters

from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt


def normalize0_255(X):
    X = (X - X.min()) / (X.max() - X.min()) * 255
    return X.round()

def matrixDivision(M, N):
    C = np.divide(M, N, out=np.zeros_like(M), where=N != 0)
    return C

dataset_path = "dataset"
results_output = "results"
image_types = ['jpg', 'JPG', 'png']
#for f in glob.glob(".*"):
#    os.remove(f)
file_names = [fn for fn in os.listdir(dataset_path)
              if any(fn.endswith(ext) for ext in image_types)]

for file in file_names:
    print("Processing file: ", file)
    L = list()

    I = imread(os.path.join(dataset_path,file))
    L.append(I)

    I = I/255
    R = I[:,:,0]
    G = I[:,:,1]
    B = I[:,:,2]
    RGB = R+G+B

    r = matrixDivision(R, RGB)
    g = matrixDivision(G, RGB)
    b = matrixDivision(B, RGB)

    # Eq (1)
    NGRDI = matrixDivision(G-R, G+R)
    NGRDI_norm = normalize0_255(NGRDI)
    L.append(NGRDI_norm)
    thr = filters.threshold_otsu(NGRDI_norm)
    NGRDI_bin =  NGRDI_norm < thr
    L.append(NGRDI_bin)


    # Eq (2)
    ExG = 2*g - r - b
    ExG_norm = normalize0_255(ExG)
    L.append(NGRDI_norm)
    thr = filters.threshold_otsu(NGRDI_norm)
    NGRDI_bin = NGRDI_norm < thr
    L.append(NGRDI_bin)


    # Eq (3)
    CIVE = 0.441*r - 0.881*g + 0.385*b + 18.78745
    CIVE_norm = normalize0_255(CIVE)
    L.append(CIVE_norm)
    thr = filters.threshold_otsu(CIVE_norm)
    CIVE_bin = CIVE_norm < thr
    L.append(CIVE_bin)

    # Eq (4)
    a=0.667
    VEG = matrixDivision(g,np.power(r,a)*np.power(b,1-a))
    VEG_norm = normalize0_255(VEG)
    L.append(VEG_norm)
    thr = filters.threshold_otsu(VEG_norm)
    VEG_bin = VEG_norm < thr
    L.append(VEG_bin)

    # Eq (5)
    ExGR = ExG -1.4*r - g
    ExGR_norm = normalize0_255(ExGR)
    L.append(ExGR_norm)
    thr = filters.threshold_otsu(ExGR_norm)
    ExGR_bin = ExGR_norm < thr
    L.append(ExGR_bin)

    # Eq (6)
    WI = matrixDivision(g-b, r-g)
    WI_norm = normalize0_255(WI)
    L.append(WI_norm)
    thr = filters.threshold_otsu(WI_norm)
    WI_bin = WI_norm < thr
    L.append(WI_bin)

    # Eq (7)
    COM = 0.25*ExG + 0.3*ExGR + 0.33*CIVE + 0.12*VEG
    COM_norm = normalize0_255(COM)
    L.append(COM_norm)
    thr = filters.threshold_otsu(COM_norm)
    COM_bin = COM_norm < thr
    L.append(COM_bin)

    # Eq (8)
    COM2 = 0.36*ExG + 0.47*CIVE + 0.17*VEG
    COM2_norm = normalize0_255(COM2)
    L.append(COM2_norm)
    thr = filters.threshold_otsu(COM2_norm)
    COM2_bin = COM2_norm < thr
    L.append(COM2_bin)

    methodsName = ['00ORIG', '01NGRDI', '02NGRDI_bin', '03ExG', '04ExG_bin', '05CIVE', '06CIVE_bin',
                   '07VEG', '08VEG_bin', '09ExGR', '10ExGR_bin', '11WI', '12WI_bin', '13COM', '14COM_bin',
                   '15COM2', '16COM2_bin']

    for i,IMG in enumerate(L):
        filename, file_ext = os.path.splitext(file)
        imsave(os.path.join(results_output,filename+'_'+methodsName[i]), IMG, 'png')