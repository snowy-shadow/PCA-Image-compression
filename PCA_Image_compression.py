import numpy as np, cv2, matplotlib.pyplot as plt
from numpy.core.fromnumeric import cumsum
from numpy import float64, linalg as la

#colors approach -- see pdf -> this is the recommended option, way faster

# read image
img_src = 'test2.jpg' # name of your image

img = cv2.cvtColor(cv2.imread(img_src), cv2.COLOR_BGR2RGB)
img_height = img.shape[0]
img_width = img.shape[1]

# split into BGR matrices
blue, green, red = cv2.split(img)

# covariance matrix
b_cov = np.cov(blue, rowvar = True, bias = True)
g_cov = np.cov(green, rowvar = True, bias = True)
r_cov = np.cov(red, rowvar = True, bias = True)

# split into eigenvals, eigenvecs
b_eigenValue, b_eigenVector = la.eigh(b_cov)
g_eigenValue, g_eigenVector = la.eigh(g_cov)
r_eigenValue, r_eigenVector = la.eigh(r_cov)

# sort eigenvalue in decreasing order
b_idx = np.argsort(b_eigenValue)[::-1]
b_eigenValue = b_eigenValue[b_idx]

g_idx = np.argsort(g_eigenValue)[::-1]
g_eigenValue = g_eigenValue[g_idx]

r_idx = np.argsort(r_eigenValue)[::-1]
r_eigenValue = r_eigenValue[r_idx]

# sort eigenvectors according to same index
b_eigenVector = b_eigenVector[:,b_idx]

g_eigenVector = g_eigenVector[:,g_idx]

r_eigenVector = b_eigenVector[:,r_idx]

# show graph
fig = plt.figure(figsize=(8, 6))
fig.suptitle("Variance explained by eigenvectors in BGR")
ax = fig.add_subplot(1, 3, 1)
plt.plot((np.cumsum(b_eigenValue) / np.sum(b_eigenValue)) * 100, marker = '.', markerfacecolor='red')
plt.title("% Variance of Blue")
plt.xlabel("Principal components")
ymajor_tics = np.arange(0, 101, 10)
yminor_ticks = np.arange(0, 101, 2)
ax.set_yticks(ymajor_tics)
ax.set_yticks(yminor_ticks, minor=True)
plt.ylabel("Cumulative Variance Explained")

ax = fig.add_subplot(1, 3, 2)
plt.plot((np.cumsum(g_eigenValue) / np.sum(g_eigenValue)) * 100, marker = '.', markerfacecolor='red')
plt.title("% Variance of Green")
plt.xlabel("Principal components")
ymajor_tics = np.arange(0, 101, 10)
yminor_ticks = np.arange(0, 101, 2)
ax.set_yticks(ymajor_tics)
ax.set_yticks(yminor_ticks, minor=True)
plt.ylabel("Cumulative Variance Explained")

ax = fig.add_subplot(1, 3, 3)
plt.plot((np.cumsum(r_eigenValue) / np.sum(r_eigenValue)) * 100, marker = '.', markerfacecolor='red')
plt.title("% Variance of Red")
plt.xlabel("Principal components")
ymajor_tics = np.arange(0, 101, 10)
yminor_ticks = np.arange(0, 101, 2)
ax.set_yticks(ymajor_tics)
ax.set_yticks(yminor_ticks, minor=True)
plt.ylabel("Cumulative Variance Explained")
plt.tight_layout()
plt.show()


def img_reduced(b_data, g_data, r_data, b_eivec, g_eivec, r_eivec, BGRpc_components):

    blue, green, red = BGRpc_components

    # pcomponents
    b_feature_vector = b_eivec[:,0:blue]
    g_feature_vector = g_eivec[:,0:green]
    r_feature_vector = r_eivec[:,0:red]

    # linear transformation
    b_compressed = np.dot(b_feature_vector.T, b_data) 
    g_compressed = np.dot(g_feature_vector.T, g_data) 
    r_compressed = np.dot(r_feature_vector.T, r_data) 
    

    # reverse PCA
    b_reconstruct = np.dot(b_feature_vector, b_compressed)
    g_reconstruct = np.dot(g_feature_vector, g_compressed) 
    r_reconstruct = np.dot(r_feature_vector, r_compressed) 

    # spliting into BGR channels
    final_blue_array = b_reconstruct.reshape(img_height, img_width)
    final_green_array = g_reconstruct.reshape(img_height, img_width)
    final_red_array = r_reconstruct.reshape(img_height, img_width)

    # combine BGR matrix
    img_reduced = cv2.merge((final_blue_array, final_green_array, final_red_array))
    # place color values between 0 - 1
    return ((img_reduced - np.min(img_reduced)) / np.ptp(img_reduced))

# image comparison
fig = plt.figure(figsize=(8, 6))
fig.suptitle("The colors approach with percent variance explained by each color")
fig.add_subplot(321)
plt.title("Original Image", size = 10)
plt.imshow(img)
fig.add_subplot(322)
plt.title("88.8% B, 85.4% G, 85.9% R", size = 10)
plt.imshow(img_reduced(blue, green, red, b_eigenVector, g_eigenVector, r_eigenVector, [3, 3, 3])) # array values controls compression amount,
fig.add_subplot(323)                                                                              # less == more compression
plt.title("96.4% B, 93.1% G, 93.7% R", size = 10)
plt.imshow(img_reduced(blue, green, red, b_eigenVector, g_eigenVector, r_eigenVector, [4, 4, 4]))
fig.add_subplot(324)
plt.title("99.9% B, 98.7% G, 98.7% R", size = 10)
plt.imshow(img_reduced(blue, green, red, b_eigenVector, g_eigenVector, r_eigenVector, [5, 5, 5]))
fig.add_subplot(325)
plt.title("100% BGR, 7 eigenvectors", size = 10)
plt.imshow(img_reduced(blue, green, red, b_eigenVector, g_eigenVector, r_eigenVector, [7, 7, 7]))
fig.add_subplot(326)
plt.title("100% BGR, all 8 eigenvectors", size = 10)
plt.imshow(img_reduced(blue, green, red, b_eigenVector, g_eigenVector, r_eigenVector, [8, 8, 8]))
plt.tight_layout()
plt.show()

# # ---------------------------------------------------------------

# # pixels approach -- see pdf 

# # read image
# img_src = 'PCA8x8PixelArt.png'

# img = cv2.cvtColor(cv2.imread(img_src), cv2.COLOR_BGR2RGB)
# img_height = img.shape[0]
# img_width = img.shape[1]

# # split into BGR matrices
# blue_array, green_array, red_array = cv2.split(img)
# # make them 1d in range 0 - 1 for easier manipulation 
# one_d_blue_array = blue_array.flatten()
# one_d_green_array = green_array.flatten() 
# one_d_red_array = red_array.flatten() 

# # new 64X3 matrix representing all pixels
# color_matrix = np.empty(shape = [0, 3], dtype = float64)
# for i in range (len(one_d_blue_array)):
#     color_matrix = np.append(color_matrix,[[one_d_blue_array[i],one_d_green_array[i],one_d_red_array[i]]], axis = 0)
# color_matrix_array = np.array(color_matrix)
# color_matrix_covariance = np.cov(color_matrix_array, rowvar = True, bias = True) 

# # split into eigenvals, eigenvecs
# eigenValue, eigenVector = la.eigh(color_matrix_covariance)

# # sort eigenvalue in decreasing order
# idx = np.argsort(eigenValue)[::-1]
# eigenValue = eigenValue[idx]

# # sort eigenvectors according to same index
# eigenVector = eigenVector[:,idx]

# # graphing PC componenets
# def CumulativeVariancePlot(Values):
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(1, 1, 1)
#     plt.plot(range(1, 65), (np.cumsum(Values)/np.sum(Values)) * 100, marker = '.', markerfacecolor = 'red')
#     plt.title("% Variance explained by the first 6 PC components")
#     plt.xlabel("Components")
#     xmajor_tics = np.arange(0, len(Values) + 1, 5)
#     xminor_ticks = np.arange(0, len(Values) + 1, 1)
#     ymajor_tics = np.arange(0, 101, 10)
#     yminor_ticks = np.arange(0, 101, 2)
#     ax.set_xticks(xmajor_tics)
#     ax.set_xticks(xminor_ticks, minor=True)
#     ax.set_yticks(ymajor_tics)
#     ax.set_yticks(yminor_ticks, minor=True)
#     plt.xlim([0.5, 6.5])
#     plt.ylabel("Cumulative Variance Explained")
#     plt.show()

# # graph
# CumulativeVariancePlot(eigenValue)

# # function for PCA
# def img_reduced (data, eigenVector, PCs):
#     # feature vector
#     feature_vector = eigenVector[:,0:PCs]

#     # linear transformation
#     compressed = np.dot (feature_vector.T, data)

#     # reverse PCA
#     reconstruct = np.dot (feature_vector, compressed)

#     # spliting into BGR channels
#     final_blue_array = reconstruct[:,0]
#     final_green_array = reconstruct[:,1]
#     final_red_array = reconstruct[:,2]

#     final_blue_array = final_blue_array.reshape(img_height,img_width)
#     final_green_array = final_green_array.reshape(img_height,img_width)
#     final_red_array = final_red_array.reshape(img_height,img_width)

#     # combine BGR matrix
#     img_reduced = cv2.merge((final_blue_array, final_green_array, final_red_array))
#     # place color values between 0 - 1
#     return (img_reduced - np.min(img_reduced)) / np.ptp(img_reduced)

# # show reduced image
# fig = plt.figure(figsize=(8, 6))
# fig.suptitle("The pixels approach")
# fig.add_subplot(321)
# plt.title("Original Image", size = 10)
# plt.imshow(img)
# fig.add_subplot(322)
# plt.title("70.7% variance, 1 eigenvector", size = 10)
# plt.imshow(img_reduced(color_matrix_array, eigenVector, 1))
# fig.add_subplot(323)
# plt.title("100% variance, 2 eigenvectors", size = 10)
# plt.imshow(img_reduced(color_matrix_array, eigenVector, 2))
# fig.add_subplot(324)
# plt.title("100% variance, 3 eigenvectors", size = 10)
# plt.imshow(img_reduced(color_matrix_array, eigenVector, 3))
# fig.add_subplot(325)
# plt.title("100% variance, 32 eigenvectors", size = 10)
# plt.imshow(img_reduced(color_matrix_array, eigenVector, 32))
# fig.add_subplot(326)
# plt.title("100% variance, all 64 eigenvectors", size = 10)
# plt.imshow(img_reduced(color_matrix_array, eigenVector, 64))
# plt.tight_layout()
# plt.show()