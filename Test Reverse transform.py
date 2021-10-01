import cv2
import numpy as np

# Forward transform
point_transformed = cv2.perspectiveTransform(point_original, trans)

# Reverse transform
inv_trans = np.linalg.pinv(trans)
round_tripped = cv2.perspectiveTransform(point_transformed, inv_trans)

# Now, round_tripped should be approximately equal to point_original