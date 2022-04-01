import os.path as osp
import sys

class Config():
    root_dir = osp.join(osp.dirname(osp.abspath(__file__)),'..')
    data_dir = osp.join(root_dir, 'data')
    
    height = 512
    width = 512

config = Config()

sys.path.insert(0, osp.join(config.root_dir))
from utils.dir import add_path
add_path(config.data_dir)
