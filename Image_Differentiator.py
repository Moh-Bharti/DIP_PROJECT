import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('original.png',0)
img2 = cv2.imread('darkend.png',0)
img3 = cv2.imread('hazed.jpg',0)
def salt_pepper():
    row,col = img1.shape
    s_vs_p = 0.5
    amount = 0.1
    out = np.copy(img1)
    # Salt mode
    num_salt = np.ceil(amount * img1.size * s_vs_p)
    coordsx = np.random.randint(0, row- 1, int(num_salt))
    coordsy = np.random.randint(0, col- 1, int(num_salt))


    for i in range(len(coordsx)):

        out[coordsx[i]][coordsy[i]] = 255

    # Pepper mode
    num_pepper = np.ceil(amount* img1.size * (1. - s_vs_p))
    coords = np.random.randint(0, row - 1, int(num_pepper))
    coordsy = np.random.randint(0, col- 1, int(num_pepper))
    for i in range(len(coordsx)):

        out[coordsx[i]][coordsy[i]] = 0


#plt.hist(out.ravel(),265,[0,256])
#plt.show()

    cv2.imshow('image',out)
    cv2.waitKey(70000)

def darkener():
    

"""print(np.mean(img1))
print(np.mean(img2))
print(np.mean(img3))
plt.figure(1)
plt.subplot(3,1,1)
plt.hist(img1.ravel(),265,[0,256])

plt.subplot(3,1,2)
plt.hist(img2.ravel(),265,[0,256])

plt.subplot(3,1,3)
plt.hist(img3.ravel(),265,[0,256])
plt.show()"""