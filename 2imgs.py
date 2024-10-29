import cv2
import numpy as np

class PanoramaStitcher:
    def __init__(self, img1, img2):
        self.left = cv2.imread(img1)
        self.right = cv2.imread(img2)
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_compute(self, image):
        kp, des = self.orb.detectAndCompute(image, None)
        return kp, des

    def match_descriptors(self, des1, des2):
        matches = self.bf.match(des1, des2)
        best = sorted(matches, key = lambda x:x.distance)[:30]
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

        M, _ = cv2.findHomography(np.float32(right_pts), np.float32(left_pts))
        dim_x = self.left.shape[1] + self.right.shape[1]
        dim_y = max(self.left.shape[0], self.right.shape[0])
        dim = (dim_x, dim_y)

        warped = cv2.warpPerspective(self.right, M, dim)
        comb = warped.copy()
        comb[0:self.left.shape[0],0:self.left.shape[1]] = self.left
        gray = cv2.cvtColor(comb, cv2.COLOR_BGR2GRAY)
        non_black_pixels_mask = gray>0
        (x, y) = np.where(non_black_pixels_mask)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped = comb[topx:bottomx+1, topy:bottomy+1]

        cv2.imwrite('stitched2imgs.jpg', cropped)

if __name__ == "__main__":
    import sys
    stitcher = PanoramaStitcher(sys.argv[1], sys.argv[2])
    stitcher.stitch()