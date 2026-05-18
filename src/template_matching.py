import cv2
import numpy as np 

def match_template(main_image,template):
    gray = cv2.cvtColor(main_image,cv2.COLOR_RGB2GRAY)

    template_gray = cv2.cvtColor(template,cv2.COLOR_RGB2GRAY)

    w,h = template_gray.shape[::-1]

    result = cv2.matchTemplate(
        gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    threshold = 0.8

    locations = zip(*((result >= threshold).nonzero()[::-1]))

    for pt in locations:
        cv2.rectangle(
            main_image,
            pt,
            (pt[0] + w,pt[1]+h),
            (0,255,0),
            2
        )
    return main_image