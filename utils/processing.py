import numpy as np
from scipy.stats import multivariate_normal

def heatmap_transform(pts, h, w):
    """
    Transforms point to heatmap 

    Args:
        pt: two-dimensional joint coordinate

    Returns:
        normalized output heatmap as 2D numpy array
    """
    maps = []
    for pt in pts:
        pos = np.dstack(np.mgrid[0:h:1,0:w:1])

        rv = multivariate_normal(mean=[pt[1],pt[0]],cov=h//8)
        
        maps.append(rv.pdf(pos))
    return np.array(maps, dtype=np.float32)

def generate_patch(img):
    """
    Transforms 
    """
    pass

