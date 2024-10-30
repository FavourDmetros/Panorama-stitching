import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from skimage.io import imread, imsave
import glob
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac
import os
import sys

class PanoramaStitcher:
    def __init__(self, image_folder):
        self.sift = cv2.SIFT_create()
        self.image_folder = image_folder
        self.load_images()
        self.num_images = len(self.imgs)
        self.tforms = [np.eye(3)]
        self.prepare_transforms()

    def load_images(self):
        imgfiles = np.sort(glob.glob(os.path.join(self.image_folder, '*.jpg')))
        self.imgs = [imread(imgfile) for imgfile in imgfiles]

    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2125 * r + 0.7154 * g + 0.0721 * b
        return gray

    def detect_and_compute(self, image):
        grayimg = self.rgb2gray(image).astype(np.uint8)
        kpt, desc = self.sift.detectAndCompute(grayimg, None)
        kpt = np.array([p.pt for p in kpt])
        return kpt, desc

    def match_features(self, desc1, desc2):
        xTy = np.inner(desc1, desc2)
        xTx = np.expand_dims(np.diag(np.inner(desc1, desc1)), axis=1)
        xTx = np.tile(xTx, (1, desc2.shape[0]))
        yTy = np.expand_dims(np.diag(np.inner(desc2, desc2)), axis=0)
        yTy = np.tile(yTy, (desc1.shape[0], 1))

        distmat = xTx + yTy - 2*xTy
        ids1 = np.argmin(distmat, axis=1)
        ids2 = np.argmin(distmat, axis=0)
        pairs = []
        for k in range(desc1.shape[0]):
            if k == ids2[ids1[k]]:
                pairs.append(np.array([k, ids1[k]]))
        pairs = np.array(pairs)

        distmat_sorted = np.sort(distmat, axis=1)
        good_pairs = []
        for i in range(pairs.shape[0]):
            k = int(pairs[i,0])
            nearestd_1 = distmat_sorted[k,0]
            nearestd_2 = distmat_sorted[k,1]
            if nearestd_1 < 0.75*nearestd_2:
                good_pairs.append(pairs[i,:])

        return np.array(good_pairs)

    def prepare_transforms(self):
        kpt, desc = self.detect_and_compute(self.imgs[0])
        for n in range(1, self.num_images):
            kpt_prev, desc_prev = kpt, desc
            kpt, desc = self.detect_and_compute(self.imgs[n])
            index_pairs = self.match_features(desc, desc_prev)
            matched_points = kpt[index_pairs[:,0],:]
            matched_points_prev = kpt_prev[index_pairs[:,1],:]
            tform, _ = ransac((matched_points, matched_points_prev), ProjectiveTransform, min_samples=4, residual_threshold=1.5, max_trials=2000)
            self.tforms.append(tform.params)

        sel = self.num_images // 2
        Tinv = np.linalg.inv(self.tforms[sel])
        for i in range(len(self.tforms)):
            self.tforms[i] = np.dot(self.tforms[i], Tinv)

    # def output_limits(self, tform, img_width, img_height):
    #     xv, yv = np.meshgrid(np.arange(1, img_width), np.arange(1, img_height))
    #     y = np.dot(tform, np.vstack((xv.flatten(), yv.flatten(), np.ones((1, xv.size)))))
    #     y_ = y[:2] / y[2]
    #     x_min, y_min = np.amin(y_, axis=0)
    #     x_max, y_max = np.amax(y_, axis=0)
    #     return [x_min, x_max], [y_min, y_max]

    def output_limits(self, tform, img_width, img_height):
        # Ensure indices start from 0 for Python
        xv, yv = np.meshgrid(np.arange(img_width), np.arange(img_height))
        
        # Applying the transformation to the points (xv, yv) and homogenizing them
        y = np.dot(tform, np.vstack((xv.flatten(), yv.flatten(), np.ones((1, xv.size)))))
        y_ = y[:2] / y[2]
        y_ = y_.T
        
        x_min, y_min = np.amin(y_, axis=0)[:2]
        x_max, y_max = np.amax(y_, axis=0)[:2]
        
        xlim = [x_min, x_max]
        ylim = [y_min, y_max]
        return xlim, ylim

    def compute_limits(self):
        xlims, ylims = [], []
        for i in range(len(self.tforms)):
            xlim, ylim = self.output_limits(self.tforms[i], self.imgs[i].shape[1], self.imgs[i].shape[0])
            xlims.append(xlim)
            ylims.append(ylim)
        return np.array(xlims), np.array(ylims)

    def imwarp(self, srcI, tform, xlims, ylims):
        xMin = np.amin(xlims[:,0])
        xMax = np.amax(xlims[:,1])
        yMin = np.amin(ylims[:,0])
        yMax = np.amax(ylims[:,1])

        width = int(np.floor(xMax - xMin))
        height = int(np.floor(yMax - yMin))

        stepx = (xMax - xMin) / (width - 1)
        stepy = (yMax - yMin) / (height - 1)

        xv, yv = np.meshgrid(np.arange(xMin, xMax + stepx, stepx), 
                             np.arange(yMin, yMax + stepy, stepy))

        pts_proj_homog = np.dot(np.linalg.inv(tform), np.vstack((xv.flatten(), yv.flatten(), np.ones((1, xv.size)))))
        pts_proj = pts_proj_homog[:2, :] / pts_proj_homog[2, :]

        xvt = pts_proj[0, :].reshape(xv.shape[0], xv.shape[1])
        yvt = pts_proj[1, :].reshape(yv.shape[0], yv.shape[1])

        warpedI = None
        if srcI.ndim == 3:
            warpedI = np.zeros((xv.shape[0], xv.shape[1], 3))
            for ch in range(3):
                warpedI[:, :, ch] = map_coordinates(srcI[:, :, ch], (yvt, xvt))
                warpedI[:, :, ch] /= np.amax(warpedI[:, :, ch])
        else:
            warpedI = map_coordinates(srcI, (yvt, xvt))
            warpedI /= np.amax(warpedI)
        
        return warpedI

    def blend(self, img0, mask0, img1):
        output = np.zeros(img0.shape)
        if img0.ndim == 3:
            for ch in range(3):
                output[:, :, ch] = (img1[:, :, ch] * (1 - mask0)) + img0[:, :, ch]
        else:
            output = (img1 * (1 - mask0)) + img0
        return output

    def stitch(self):
        xlims, ylims = self.compute_limits()
        panorama = None

        for n in range(len(self.tforms)):
            img = self.imgs[n]
            tform = self.tforms[n]
            imgt = self.imwarp(img, tform, xlims, ylims)
            mask = np.ones(img.shape[:2])
            maskt = self.imwarp(mask, tform, xlims, ylims)

            if panorama is not None:
                panorama = self.blend(imgt, maskt, panorama)
            else:
                panorama = np.zeros((imgt.shape[0], imgt.shape[1], 3))
                panorama = self.blend(imgt, maskt, panorama)
        
        return panorama

    def save_panorama(self, output_path):
        panorama = self.stitch()
        imsave(output_path, (panorama * 255).astype(np.uint8))
        print(f"Panorama saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python nimages.py <folder_with_imgs>")
        sys.exit(1)

    image_folder = sys.argv[1]
    output_path = os.path.join(image_folder, "panorama.jpg")
    
    stitcher = PanoramaStitcher(image_folder)
    stitcher.save_panorama(output_path)
