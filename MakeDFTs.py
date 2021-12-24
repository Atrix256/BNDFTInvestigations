
import imageio
import numpy as np
from skimage import exposure
from PIL import Image

filePatterns=[
    "blue2d_64x64",
    "blue3d_64x64x64",
    "blue2dx1d_64x64x64"
]

# Load the images and write out DFTs
for filePattern in filePatterns:
    images = []

    # Make the XY DFTs
    avgDFT = []
    for i in range(64):
        fileName = "source/" + filePattern + "_" + str(i) + ".png"
        image = imageio.imread(fileName).astype(float)
        images.append(image.copy())
        
        img_c2 = np.fft.fft2(image)
        img_c3 = np.fft.fftshift(img_c2)
        img_c4 = np.abs(img_c3);

        if i == 0:
            avgDFT = img_c4 / 64.0
        else:
            avgDFT = avgDFT + img_c4 / 64.0

        if i == 0:
            imageio.imwrite("out/" + filePattern + ".mag2d." + str(i) + ".xy.png", np.log(1+img_c4))            

    imageio.imwrite("out/" + filePattern + ".mag2d.combined.xy.png", np.log(1+avgDFT))

    # Rotate the images 90 degress to make the XZ DFTs
    imagesRot = np.rot90(images, axes=(0,1))

    # Make the XZ DFTs
    avgDFT = []
    for i in range(imagesRot.shape[2]):
        image=imagesRot[i]

        img_c2 = np.fft.fft2(image)
        img_c3 = np.fft.fftshift(img_c2)
        img_c4 = np.abs(img_c3);

        if i == 0:
            avgDFT = img_c4 / 64.0
        else:
            avgDFT = avgDFT + img_c4 / 64.0

        if i == 0:
            imageio.imwrite("out/" + filePattern + ".mag2d." + str(i) + ".xz.png", np.log(1+img_c4))            

    imageio.imwrite("out/" + filePattern + ".mag2d.combined.xz.png", np.log(1+avgDFT))

    # Make a XY sliced 3D DFT
    imgs_c2 = np.fft.fftn(images)
    imgs_c3 = np.fft.fftshift(imgs_c2)
    imgs_c4 = np.abs(imgs_c3)
    imgs_c5 = np.log(1+imgs_c4)
    for i in range(imgs_c5.shape[2]):
        image=imgs_c5[i]
        imageio.imwrite("out/" + filePattern + ".mag3d.xy." + str(i) + ".png", image)        
    
    # Make a XZ sliced 3D DFT
    imgs_c2 = np.fft.fftn(imagesRot)
    imgs_c3 = np.fft.fftshift(imgs_c2)
    imgs_c4 = np.abs(imgs_c3)
    imgs_c5 = np.log(1+imgs_c4)
    for i in range(imgs_c5.shape[2]):
        image=imgs_c5[i]
        imageio.imwrite("out/" + filePattern + ".mag3d.xz." + str(i) + ".png", image)        

