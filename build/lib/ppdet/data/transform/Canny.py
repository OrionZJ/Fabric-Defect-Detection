import cv2

class Canny(object):
    def __init__(self, threshold1, threshold2):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def __call__(self, data):
        image = data['image']
        canny_image = cv2.Canny(image, self.threshold1, self.threshold2)
        data['canny_image'] = canny_image
        return data