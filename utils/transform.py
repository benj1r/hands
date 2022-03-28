import numpy as np

def global2local(coord, R, T):
    """
    Transforms world-space joint coordinate to camera-space joint coordinate

    Args:
        coord: joint coordinate numpy array
        R: camera rotation matrix
        T: transpose camera position matrix
    
    Returns:
        joint coordinate in camera-space
    """
    pass


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
    pass


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
    pass


