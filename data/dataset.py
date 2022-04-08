import sys
import os.path as osp
import torch
import numpy as np
import json
import cv2
from pycocotools.coco import COCO

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from config import config 
from utils.transform import global2local, local2pixel, pixel2local
from utils.processing import heatmap_transform, generate_patch, trans_pt

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode, dataset):
        self.transform = transform
        self.mode = mode
        
        self.curr_dir = osp.join(osp.dirname(osp.abspath(__file__)))
        self.ann_path = osp.join(self.curr_dir, 'dataset/annotation')
        self.img_path = osp.join(self.curr_dir, 'dataset/images')
    
        self.joint_num = 21
        self.joint_type = {
                'right': np.arange(0, self.joint_num),
                'left' : np.arange(self.joint_num, self.joint_num * 2)
                }

        self.dlist = []
        
        db = COCO(osp.join(self.ann_path, dataset + '_' + self.mode + '_data.json'))
        
        with open(osp.join(self.ann_path, dataset + '_' + self.mode + '_camera.json')) as f:
            cameras = json.load(f)

        with open(osp.join(self.ann_path, dataset + '_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)
        
        for id in db.anns.keys():
            ann = db.anns[id]
            img_id = ann['image_id']
            img = db.loadImgs(img_id)[0]
            capture_id = img['capture']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.img_path, self.mode, img['file_name'])

            campos = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32)
            camrot = np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32)
            princpt = np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            joint_cam = global2local(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
            joint_img = local2pixel(joint_cam, focal, princpt)[:,:2]
            
            bbox = np.array(ann['bbox'],dtype=np.float32)
            

            cam_param = {
                    'focal': focal,
                    'princpt': princpt
                    }
            joint = {
                    'cam_coord': joint_cam,
                    'img_coord': joint_img,
                    }
            data = {
                    'img_path': img_path,
                    'cam_param': cam_param,
                    'joint': joint,
                    'bbox': bbox
                    }
            
            self.dlist.append(data)

    def __len__(self):
        return len(self.dlist)
    
    
    def __getitem__(self, idx):    
        data = self.dlist[idx]

        # inputs
        img_path = data['img_path']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # targets
        joint = data['joint']
        joint_cam = joint['cam_coord']
        joint_img = joint['img_coord']
        bbox = data['bbox']
        
        img, trans = generate_patch(img, bbox)
            
        for i in range(len(joint_img)):
            joint_img[i,:2] = trans_pt(joint_img[i,:2], trans)
        
        img = self.transform(img.astype(np.float32)) / 255.0
        joint = self.transform(joint_img.astype(np.float32))
        joint = torch.squeeze(joint)

        joint = heatmap_transform(joint, config.height, config.width)
        
        inputs = img
        targets = joint
        
        return inputs.squeeze(), targets.squeeze()


