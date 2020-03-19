import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread('apple1.jpg',1)

print(np.shape(img))

def background_remover(img):
    n = len(img)
    m = len(img[0])

    im = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            cell = img[i][j]

            rgb = np.max(cell)
            if cell[0]==rgb:
                im[i][j]=255

            if cell[0]>10 or cell[1]<60 or cell[2]<20:
                im[i][j]=255
    return im

def CHT(img,r):
    n = len(img)
    m = len(img[0])
    sine = dict()
    cosine = dict()
    circles = []
    for ang in range(0,360):
        sine[ang] = np.sin(ang*np.pi/180)
        cosine[ang] = np.cos(ang*np.pi/180)

    for rad in range(r):
        cells = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                if img[i][j]==255:
                    for ang in range(0,360):
                        b = i-int(rad*sine[ang])
                        a = j - int(rad * cosine[ang])
                        if a>=0 and a<m and b>=0 and b<n:
                            cells[b][a]+=1
        print('radius',rad)
        cell_max = np.amax(cells)
        print("max value",cell_max)

        if (cell_max>150):
            print("Detecting")
            cells[cells<150]=0

            for k in range(n):
                for l in range(m):
                    if(k>0 and l>0 and k<n-1 and l<m-1 and cells[k][l]>=100):
                        mat = cells[k-1:k+1][l-1:l+1]
                        avg = np.float(np.mean(mat))
                        if (avg>=33):
                            circles.append((k,l,rad))
                            cells[k:k+5][l:l+7] = 0

    return circles




if __name__=='__main__':
    #cv2.imshow('image',img)
    #cv2.waitKey(10000)
    image1 = background_remover(img)
    image2 = Image.fromarray(image1,mode=None)
    image3 = cv2.medianBlur(np.float32(image2),5)
    circles = CHT(image3,3)
    for coord in circles:
        cv2.circle(img,(coord[1],coord[0]),coord[2],(0,255,0),1)
        cv2.rectangle(img,(coord[1]-2,coord[0]-2),(coord[1]-2,coord[2]-2),(0,0,255),3)
    print(circles)

    cv2.imshow('Circle Detected',img)
    cv2.waitKey(10000)
    #--------------------------------------------------------------------
    """conts, h = cv2.findContours(image3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, conts, -1, (255, 0, 0), 1)
    for i in range(len(conts)):
        x, y, w, h = cv2.boundingRect(conts[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, str(i + 1), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))"""


    #-------------------------------------------------------------------------




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