import numpy as np
from scipy.stats import multivariate_normal
import cv2

from config import config

def heatmap_transform(pts, h, w):
    """
    Transforms point to heatmap 

    Args:
        pts: set of two-dimensional joint coordinates
        h: maximum coordinate height
        w: maximum coordinate width

    Returns:
        Normalized output heatmap as 2D numpy array
    """

    maps = []
    for pt in pts:
        pos = np.dstack(np.mgrid[0:h:1,0:w:1])

        rv = multivariate_normal(mean=[pt[1],pt[0]],cov=h//4)
        hm = rv.pdf(pos)
        
        maps.append(hm)
    
    return np.array(maps, dtype=np.float32)


def generate_patch(img, bbox):
    """
    Transforms an image towards the center of the joint bounding box
    
    Args:
        img: numpy array representing 2D input image
        bbox: bounding box of joint coordinates

    Returns:
        patch: transformed image
        trans: affine transformation matrix
    """

    h, w = img.shape
    
    bbox_center_x = float(bbox[0] + 0.5*bbox[2])
    bbox_center_y = float(bbox[1] + 0.5*bbox[3])

    bbox_width = float(bbox[2])
    bbox_height = float(bbox[3])

    bbox_center = np.array([bbox_center_x, bbox_center_y], dtype=np.float32)
    bbox_down = np.array([0,bbox_height * 0.5], dtype=np.float32)
    bbox_right = np.array([bbox_width * 0.5, 0], dtype=np.float32)
    
    # output image shape
    dst_width = config.width
    dst_height = config.height
    
    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_down = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_right = np.array([dst_width * 0.5, 0], dtype=np.float32)
    
    bb = np.zeros((3,2), dtype=np.float32)
    bb[0, :] = bbox_center
    bb[1, :] = bbox_center + bbox_down
    bb[2, :] = bbox_center + bbox_right
    
    dst = np.zeros((3,2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_down
    dst[2, :] = dst_right + dst_right
    
    trans = cv2.getAffineTransform(np.float32(bb), np.float32(dst))

    trans = trans.astype(np.float32)

    patch = cv2.warpAffine(img, trans, (dst_width, dst_height))

    return patch, trans


def trans_pt(pt, trans):
    src_pt = np.array([pt[0], pt[1], 1.0]).T
    dst_pt = np.dot(trans, src_pt)

    return dst_pt[0:2]
