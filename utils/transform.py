import numpy as np

def global2local(coord, R, T):
    """
    Transforms world-space joint coordinate to camera-space joint coordinate

    Args:
        coord: transpose joint coordinate numpy array
        R: camera rotation matrix
        T: camera position matrix
    
    Returns:
        joint coordinate in camera-space
    """
    
    coord = np.dot(R, coord - T)
    return coord

def local2pixel(coord, f, c):
    """
    Transforms camera-space joint coordinate to 2d joint coordinate in pixel-space

    Args:
        coord: joint coordinate numpy array
        f: camera focal length
        c: camera principle point
    
    Returns:
        joint coordinate in pixel-space
    """
    
    esp = 1e-8

    x = coord[:, 0] / (coord[:, 2] + esp) * f[0] + c[0]
    
    y = coord[:, 1] / (coord[:, 2] + esp) * f[1] + c[1]
    
    z = coord[:, 2]
    
    return np.concatenate((x[:,None],y[:,None],z[:,None]),1)


def pixel2local(coord, f, c):
    """
    Transforms 2D pixel-space joint coordinate to camera-space joint coordinate

    Args:
        coord: joint coordinate numpy array
        f: camera focal length
        c: camera principle point
    
    Returns:
        joint coordinate in camera-space
    """
    x = (coord[:,0] - c[0]) / f[0] * coord[:,2]
    
    y = (coord[:,1] - c[1]) / f[1] * coord[:,2]
    
    z = coord[:, 2]
    
    return np.concatenate((x[:,None],y[:,None], z[:,None]),1)

