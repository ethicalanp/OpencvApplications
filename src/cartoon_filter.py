import cv2

def cartoon_filter(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray,5)

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    color = cv2.bilateralFilter(image,
                                9,
                                250,
                                300)
    cartoon = cv2.bitwise_and(color,
                              color,
                              mask=edges)
    return cartoon
