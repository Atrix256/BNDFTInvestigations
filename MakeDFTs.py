
import imageio
import numpy as np
from skimage import exposure
from PIL import Image, ImageFont, ImageDraw 
import sys

filePatterns=[
    "blue2d_64x64",
    "blue3d_64x64x64",
    "blue2dx1d_64x64x64"
]

label_font = ImageFont.truetype('arial.ttf', 10)

# Combine the first three blue2d_64x64 images into a vec2 and a vec3, so we can compare it to a real vec3
blue2D_RGB = np.empty([64, 64, 3], dtype=np.uint8)
blue2D_RGB[:,:,0] = imageio.imread("source/blue2d_64x64_0.png")
blue2D_RGB[:,:,1] = imageio.imread("source/blue2d_64x64_1.png")
blue2D_RGB[:,:,2] = 0
imageio.imwrite("out/blue2d_64x64_RG.png", blue2D_RGB)
blue2D_RGB[:,:,2] = imageio.imread("source/blue2d_64x64_2.png")
imageio.imwrite("out/blue2d_64x64_RGB.png", blue2D_RGB)

blue2D_RGB = blue2D_RGB.astype(float) / 255.0

# make channel wise dft of the vec2 and vec3 images we made
blue2D_R_DFT = np.fft.fft2(blue2D_RGB[:,:,0])
blue2D_R_DFT2 = np.fft.fftshift(blue2D_R_DFT)
blue2D_R_DFT3 = np.abs(blue2D_R_DFT2);
blue2D_R_DFT4 = np.log(1+blue2D_R_DFT3)
blue2D_R_DFT5 = blue2D_R_DFT4 / max(1, np.amax(blue2D_R_DFT4))

blue2D_G_DFT = np.fft.fft2(blue2D_RGB[:,:,1])
blue2D_G_DFT2 = np.fft.fftshift(blue2D_G_DFT)
blue2D_G_DFT3 = np.abs(blue2D_G_DFT2);
blue2D_G_DFT4 = np.log(1+blue2D_G_DFT3)
blue2D_G_DFT5 = blue2D_G_DFT4 / max(1, np.amax(blue2D_G_DFT4))

blue2D_B_DFT = np.fft.fft2(blue2D_RGB[:,:,2])
blue2D_B_DFT2 = np.fft.fftshift(blue2D_B_DFT)
blue2D_B_DFT3 = np.abs(blue2D_B_DFT2);
blue2D_B_DFT4 = np.log(1+blue2D_B_DFT3)
blue2D_B_DFT5 = blue2D_B_DFT4 / max(1, np.amax(blue2D_B_DFT4))

blue2D_RGB[:,:,0] = blue2D_R_DFT5
blue2D_RGB[:,:,1] = blue2D_G_DFT5
blue2D_RGB[:,:,2] = 0
imageio.imwrite("out/blue2d_64x64_RG.mag.png", (blue2D_RGB * 255.0).astype(np.uint8))
blue2D_RGB[:,:,2] = blue2D_B_DFT5
imageio.imwrite("out/blue2d_64x64_RGB.mag.png", (blue2D_RGB * 255.0).astype(np.uint8))

# make a combined DFT
blue2D_R_DFT = np.fft.fft2(blue2D_RGB[:,:,0])
blue2D_R_DFT2 = np.fft.fftshift(blue2D_R_DFT)

blue2D_G_DFT = np.fft.fft2(blue2D_RGB[:,:,1])
blue2D_G_DFT2 = np.fft.fftshift(blue2D_G_DFT)

blue2D_B_DFT = np.fft.fft2(blue2D_RGB[:,:,2])
blue2D_B_DFT2 = np.fft.fftshift(blue2D_B_DFT)

blue2D_DFT2 = np.sqrt(blue2D_R_DFT2*blue2D_R_DFT2 + blue2D_G_DFT2*blue2D_G_DFT2)
blue2D_DFT3 = np.abs(blue2D_DFT2);
blue2D_DFT4 = np.log(1+blue2D_DFT3)
blue2D_DFT5 = blue2D_DFT4 / max(1, np.amax(blue2D_DFT4))
imageio.imwrite("out/blue2d_64x64_RG.mag.combined.png", (blue2D_DFT5 * 255.0).astype(np.uint8))

blue2D_DFT2 = np.sqrt(blue2D_R_DFT2*blue2D_R_DFT2 + blue2D_G_DFT2*blue2D_G_DFT2 + blue2D_B_DFT2*blue2D_B_DFT2)
blue2D_DFT3 = np.abs(blue2D_DFT2);
blue2D_DFT4 = np.log(1+blue2D_DFT3)
blue2D_DFT5 = blue2D_DFT4 / max(1, np.amax(blue2D_DFT4))
imageio.imwrite("out/blue2d_64x64_RGB.mag.combined.png", (blue2D_DFT5 * 255.0).astype(np.uint8))

# make channel wise dft of the actual vec2 blue noise image
imageIn = imageio.imread("source/blue2d_vec2_64x64.png")
blue2D_RGB[:,:,0] = imageIn[:,:,0].astype(float) / 255.0
blue2D_RGB[:,:,1] = imageIn[:,:,1].astype(float) / 255.0
blue2D_RGB[:,:,2] = 0

blue2D_R_DFT = np.fft.fft2(blue2D_RGB[:,:,0])
blue2D_R_DFT2 = np.fft.fftshift(blue2D_R_DFT)
blue2D_R_DFT3 = np.abs(blue2D_R_DFT2);
blue2D_R_DFT4 = np.log(1+blue2D_R_DFT3)
blue2D_R_DFT5 = blue2D_R_DFT4 / max(1, np.amax(blue2D_R_DFT4))

blue2D_G_DFT = np.fft.fft2(blue2D_RGB[:,:,1])
blue2D_G_DFT2 = np.fft.fftshift(blue2D_G_DFT)
blue2D_G_DFT3 = np.abs(blue2D_G_DFT2);
blue2D_G_DFT4 = np.log(1+blue2D_G_DFT3)
blue2D_G_DFT5 = blue2D_G_DFT4 / max(1, np.amax(blue2D_G_DFT4))

blue2D_RGB[:,:,0] = blue2D_R_DFT5
blue2D_RGB[:,:,1] = blue2D_G_DFT5
blue2D_RGB[:,:,2] = 0
imageio.imwrite("out/blue2d_vec2_64x64.mag.png", (blue2D_RGB * 255.0).astype(np.uint8))

# make channel wise dft of the actual vec3 blue noise image
imageIn = imageio.imread("source/blue2d_vec3_64x64.png")
blue2D_RGB[:,:,0] = imageIn[:,:,0].astype(float) / 255.0
blue2D_RGB[:,:,1] = imageIn[:,:,1].astype(float) / 255.0
blue2D_RGB[:,:,2] = imageIn[:,:,2].astype(float) / 255.0

blue2D_R_DFT = np.fft.fft2(blue2D_RGB[:,:,0])
blue2D_R_DFT2 = np.fft.fftshift(blue2D_R_DFT)
blue2D_R_DFT3 = np.abs(blue2D_R_DFT2);
blue2D_R_DFT4 = np.log(1+blue2D_R_DFT3)
blue2D_R_DFT5 = blue2D_R_DFT4 / max(1, np.amax(blue2D_R_DFT4))

blue2D_G_DFT = np.fft.fft2(blue2D_RGB[:,:,1])
blue2D_G_DFT2 = np.fft.fftshift(blue2D_G_DFT)
blue2D_G_DFT3 = np.abs(blue2D_G_DFT2);
blue2D_G_DFT4 = np.log(1+blue2D_G_DFT3)
blue2D_G_DFT5 = blue2D_G_DFT4 / max(1, np.amax(blue2D_G_DFT4))

blue2D_B_DFT = np.fft.fft2(blue2D_RGB[:,:,2])
blue2D_B_DFT2 = np.fft.fftshift(blue2D_B_DFT)
blue2D_B_DFT3 = np.abs(blue2D_B_DFT2);
blue2D_B_DFT4 = np.log(1+blue2D_B_DFT3)
blue2D_B_DFT5 = blue2D_B_DFT4 / max(1, np.amax(blue2D_B_DFT4))

blue2D_RGB[:,:,0] = blue2D_R_DFT5
blue2D_RGB[:,:,1] = blue2D_G_DFT5
blue2D_RGB[:,:,2] = blue2D_B_DFT5
imageio.imwrite("out/blue2d_vec3_64x64.mag.png", (blue2D_RGB * 255.0).astype(np.uint8))

# make a combined DFT of the actual vec2 blue noise image
imageIn = imageio.imread("source/blue2d_vec2_64x64.png")
blue2D_RGB[:,:,0] = imageIn[:,:,0].astype(float) / 255.0
blue2D_RGB[:,:,1] = imageIn[:,:,1].astype(float) / 255.0
blue2D_RGB[:,:,2] = 0

blue2D_R_DFT = np.fft.fft2(blue2D_RGB[:,:,0])
blue2D_R_DFT2 = np.fft.fftshift(blue2D_R_DFT)

blue2D_G_DFT = np.fft.fft2(blue2D_RGB[:,:,1])
blue2D_G_DFT2 = np.fft.fftshift(blue2D_G_DFT)

blue2D_DFT2 = np.sqrt(blue2D_R_DFT2*blue2D_R_DFT2 + blue2D_G_DFT2*blue2D_G_DFT2)
blue2D_DFT3 = np.abs(blue2D_DFT2);
blue2D_DFT4 = np.log(1+blue2D_DFT3)
blue2D_DFT5 = blue2D_DFT4 / max(1, np.amax(blue2D_DFT4))
imageio.imwrite("out/blue2d_vec2_64x64.mag.combined.png", (blue2D_DFT5 * 255.0).astype(np.uint8))

# make a combined DFT of the actual vec3 blue noise image
imageIn = imageio.imread("source/blue2d_vec3_64x64.png")
blue2D_RGB[:,:,0] = imageIn[:,:,0].astype(float) / 255.0
blue2D_RGB[:,:,1] = imageIn[:,:,1].astype(float) / 255.0
blue2D_RGB[:,:,2] = imageIn[:,:,2].astype(float) / 255.0

blue2D_R_DFT = np.fft.fft2(blue2D_RGB[:,:,0])
blue2D_R_DFT2 = np.fft.fftshift(blue2D_R_DFT)

blue2D_G_DFT = np.fft.fft2(blue2D_RGB[:,:,1])
blue2D_G_DFT2 = np.fft.fftshift(blue2D_G_DFT)

blue2D_B_DFT = np.fft.fft2(blue2D_RGB[:,:,2])
blue2D_B_DFT2 = np.fft.fftshift(blue2D_B_DFT)

blue2D_DFT2 = np.sqrt(blue2D_R_DFT2*blue2D_R_DFT2 + blue2D_G_DFT2*blue2D_G_DFT2 + blue2D_B_DFT2*blue2D_B_DFT2)
blue2D_DFT3 = np.abs(blue2D_DFT2);
blue2D_DFT4 = np.log(1+blue2D_DFT3)
blue2D_DFT5 = blue2D_DFT4 / max(1, np.amax(blue2D_DFT4))
imageio.imwrite("out/blue2d_vec3_64x64.mag.combined.png", (blue2D_DFT5 * 255.0).astype(np.uint8))

# Load the images and write out DFTs
for filePattern in filePatterns:
    images = []

    # Make the XY DFTs
    avgDFT = []
    for i in range(64):
        fileName = "source/" + filePattern + "_" + str(i) + ".png"
        image = imageio.imread(fileName).astype(float) / 255.0

        images.append(image.copy())
        
        img_c2 = np.fft.fft2(image)
        img_c3 = np.fft.fftshift(img_c2)
        img_c4 = np.abs(img_c3);

        if i == 0:
            avgDFT = img_c4 / 64.0
        else:
            avgDFT = avgDFT + img_c4 / 64.0

        if i == 0:
            img_c5 = np.log(1+img_c4)
            img_c6 = img_c5 / max(1, np.amax(img_c5))
            imageio.imwrite("out/" + filePattern + ".mag2d." + str(i) + ".xy.png", (img_c6 * 255.0).astype(np.uint8))

    avgDFT2 = np.log(1+avgDFT)
    avgDFT3 = avgDFT2 / max(1, np.amax(avgDFT2))
    imageio.imwrite("out/" + filePattern + ".mag2d.avg.xy.png", (avgDFT3 * 255.0).astype(np.uint8))

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
            img_c5 = np.log(1+img_c4)
            img_c6 = img_c5 / max(1, np.amax(img_c5))
            imageio.imwrite("out/" + filePattern + ".mag2d." + str(i) + ".xz.png", (img_c6 * 255.0).astype(np.uint8))

    avgDFT2 = np.log(1+avgDFT)
    avgDFT3 = avgDFT2 / max(1, np.amax(avgDFT2))
    imageio.imwrite("out/" + filePattern + ".mag2d.avg.xz.png", (avgDFT3 * 255.0).astype(np.uint8))

    # Make a XY sliced 3D DFT
    imgs_c2 = np.fft.fftn(images)
    imgs_c3 = np.fft.fftshift(imgs_c2)
    imgs_c4 = np.abs(imgs_c3)
    imgs_c5 = np.log(1+imgs_c4)
    imgs_c6 = imgs_c5 / max(1, np.amax(imgs_c5))
    for i in range(imgs_c6.shape[2]):
        image=imgs_c6[i]
        imageio.imwrite("out/" + filePattern + ".mag3d.xy." + str(i) + ".png", (image * 255.0).astype(np.uint8))        
    
    # Make a XZ sliced 3D DFT
    imgs_c2 = np.fft.fftn(imagesRot)
    imgs_c3 = np.fft.fftshift(imgs_c2)
    imgs_c4 = np.abs(imgs_c3)
    imgs_c5 = np.log(1+imgs_c4)
    imgs_c6 = imgs_c5 / max(1, np.amax(imgs_c5))
    for i in range(imgs_c6.shape[2]):
        image=imgs_c6[i]
        imageio.imwrite("out/" + filePattern + ".mag3d.xz." + str(i) + ".png", (image * 255.0).astype(np.uint8))        

# Make a combined image showing 2d and 3d noise of true vectors vs combining scalar textures
im11 = Image.open("out/blue2d_64x64_RG.png")
im21 = Image.open("out/blue2d_64x64_RG.mag.png")
im31 = Image.open("out/blue2d_64x64_RG.mag.combined.png")

im12 = Image.open("source/blue2d_vec2_64x64.png")
im22 = Image.open("out/blue2d_vec2_64x64.mag.png")
im32 = Image.open("out/blue2d_vec2_64x64.mag.combined.png")

im13 = Image.open("out/blue2d_64x64_RGB.png")
im23 = Image.open("out/blue2d_64x64_RGB.mag.png")
im33 = Image.open("out/blue2d_64x64_RGB.mag.combined.png")

im14 = Image.open("source/blue2d_vec3_64x64.png")
im24 = Image.open("out/blue2d_vec3_64x64.mag.png")
im34 = Image.open("out/blue2d_vec3_64x64.mag.combined.png")

imout = Image.new('RGB',(3*im11.size[0] + 6, 4*im11.size[1] + 9), (255, 255, 255))
imout.paste(im11, (im11.size[0]*0, im11.size[1]*0))
imout.paste(im21, (im11.size[0]*1+3, im11.size[1]*0))
imout.paste(im31, (im11.size[0]*2+6, im11.size[1]*0))
imout.paste(im12, (im11.size[0]*0, im11.size[1]*1+3))
imout.paste(im22, (im11.size[0]*1+3, im11.size[1]*1+3))
imout.paste(im32, (im11.size[0]*2+6, im11.size[1]*1+3))
imout.paste(im13, (im11.size[0]*0, im11.size[1]*2+6))
imout.paste(im23, (im11.size[0]*1+3, im11.size[1]*2+6))
imout.paste(im33, (im11.size[0]*2+6, im11.size[1]*2+6))
imout.paste(im14, (im11.size[0]*0, im11.size[1]*3+9))
imout.paste(im24, (im11.size[0]*1+3, im11.size[1]*3+9))
imout.paste(im34, (im11.size[0]*2+6, im11.size[1]*3+9))

imout2 = Image.new('RGB',(imout.size[0] + 75, imout.size[1] + 25), (255, 255, 255))
imout2.paste(imout, (70, 20))

imout2_editable = ImageDraw.Draw(imout2)
imout2_editable.text((5,52), "Scalar RG", (0, 0, 0), font=label_font)
imout2_editable.text((5,119), "Vec2 RG", (0, 0, 0), font=label_font)
imout2_editable.text((5,186), "Scalar RGB", (0, 0, 0), font=label_font)
imout2_editable.text((5,253), "Vec3 RGB", (0, 0, 0), font=label_font)

imout2_editable.text((85,5), "Texture", (0, 0, 0), font=label_font)
imout2_editable.text((142,5), "DFT Single", (0, 0, 0), font=label_font)
imout2_editable.text((211,5), "Combined", (0, 0, 0), font=label_font)

imout2.save("out/_vec23.png")
