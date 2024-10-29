import cv2
import numpy as np

class PanoramaStitcher:
    def __init__(self, img1, img2):
        self.left = cv2.imread(img1)
        self.right = cv2.imread(img2)
        self.sift = cv2.SIFT_create()  # Use SIFT for better feature detection
        self.bf = cv2.BFMatcher()

    def detect_and_compute(self, image):
        kp, des = self.sift.detectAndCompute(image, None)
        return kp, des

    def match_descriptors(self, des1, des2):
        matches = self.bf.knnMatch(des1, des2, k=2)
        # Apply ratio test as per Lowe's paper
        best = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                best.append(m)
        return best

    def find_points(self, matches, kp1, kp2):
        left_pts = []
        right_pts = []
        for m in matches:
            l = kp1[m.queryIdx].pt
            r = kp2[m.trainIdx].pt
            left_pts.append(l)
            right_pts.append(r)
        return left_pts, right_pts

    def stitch(self):
        kp_left, des_left = self.detect_and_compute(self.left)
        kp_right, des_right = self.detect_and_compute(self.right)
        best_matches = self.match_descriptors(des_left, des_right)
        left_pts, right_pts = self.find_points(best_matches, kp_left, kp_right)

        M, mask = cv2.findHomography(np.float32(right_pts), np.float32(left_pts), cv2.RANSAC, 5.0)
        dim_x = self.left.shape[1] + self.right.shape[1]
        dim_y = max(self.left.shape[0], self.right.shape[0])
        result = cv2.warpPerspective(self.right, M, (dim_x, dim_y))
        result[:self.left.shape[0], :self.left.shape[1]] = self.left

        # Use a multi-band blending technique for seamless transition (optional)
        # This step needs an additional implementation of a blending function

        cv2.imwrite('stitched2imgs_2.jpg', result)

if __name__ == "__main__":
    import sys
    stitcher = PanoramaStitcher(sys.argv[1], sys.argv[2])
    stitcher.stitch()