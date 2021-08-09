import numpy as np
import cv2

center = np.array([[[0, 0]]])
M = cv2.getRotationMatrix2D(
    (
        112,
        112,
    ),
    45,
    1,
)
new_center = cv2.transform(center, M)
new_center = np.transpose(new_center[0])
image = np.zeros((224,224))
image[0, 0] = 255
image[0, 223] = 128
print(center)
print(new_center)
cv2.imshow('test', image)
cv2.waitKey(0)
cv2.destroyAllWindows()