import cv2

def apply_gradients(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0)

    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1)

    laplacian = cv2.Laplacian(gray,cv2.CV_64F)

    return sobelx,sobely,laplacian