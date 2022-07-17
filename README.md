# To IB
## *_This is my IB math EE research submited for June 2022, so do not mark it for plagerism thank you_*

# PCA Image compression _(not SVD)_ <br/>
The script uses PCA (princial component analysis) to perform image compression. PCA is not limited to image compression, it can be used to compress a variety of multivariable datasets.
- PCA is very similar to SVD in principal but the two uses two different techniques in its calculation to perform compression. 
- This is a <u>_lossy_</u> compression technique. Data will be lost when compression is performed
- see the PDF file for the maths

PCA breaks down the matrix(which often is a data table with multiple variables) into Eigenvectors and Values known as Eigen decomposition. These Eigenvectors are called components

Given this 8x8 image(enlarged because its only 64 pixels) <br/>
![8x8 Pixel Art](./misc/PCA8x8PixelArt.png) <br/>
![8x8 Pixel Art Enlarged](./misc/8x8PixelGraph.png) <br/> <br/>
A matrix can be formed by taking its RBG data values and place them into a matrix. Then finding its PCs and ording their eigenvalues from smallest to largest -> <br/>
![BGR PCs graph](./misc/BGR%20PCs%20graph.png)<br/><br/>
and then compression can be applied by selecting the amount of variance(the amount of eigen vectors) to retain. The more variance retained the better the image quality and also larger the file size.<br/>
- usually aiming to retain an arbitrary 95% variance

Below is the result of compression<br/>
![compressed images](./misc/Colors%20approach.png)<br/><br/>
## A different approach to forming the matrix was also discussed in the paper (very slow)
Here is the result<br/>
![64x3 cumulative graph](./misc/64X3_components_cumulative.png)<br/>
![64x3 cumulative graph zoomed](./misc/64x3%20PCs.png)
![64x3 compression result](./misc//Pixel%20Approach.png)


# Required environment
- numpy + matplotlib
- installing anaconda will make the process so much easier

