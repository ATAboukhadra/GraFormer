import numpy as np
import cv2
import io
import os
import torch.utils.data as data
import pickle
from PIL import Image

class Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', seq_length=1, mesh=False):

        self.root = root 
        self.load_set = load_set  # 'train','val','test'
        self.seq_length = seq_length
        self.num_classes = 29
        self.mesh = mesh
        # self.rcnn_dict = pickle.load(open('rcnn_outputs_%s.pkl'%self.load_set, 'rb'))
        # print(rcnn_dict)
        self.images = np.load(os.path.join(root, 'images-%s.npy'%self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        if mesh:
            self.mesh2d = np.load(os.path.join(root, 'mesh2d-%s.npy'%self.load_set))[:, :778]
            self.mesh3d = np.load(os.path.join(root, 'mesh3d-%s.npy'%self.load_set))[:, :778]
            # print(self.mesh2d.shape)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        
        if self.mesh:
            point2d = self.mesh2d[index][:778]
            point3d = self.mesh3d[index][:778]

        else:    
            # path = self.images[index]
            point2d = self.points2d[index]
            # point2d_rcnn = self.rcnn_dict[path]
            point3d = self.points3d[index]

        if self.seq_length > 1:
            frame_num = int(self.images[index].split('/')[-1].split('.')[0])
            point2d, point3d = self.create_sequence(index, frame_num)
        
        return point2d, point3d#, point2d_rcnn

    def __len__(self):
        return len(self.images)

    def create_sequence(self, index, frame_num):

        point2d_seq = np.zeros((self.seq_length, self.num_classes, 2))
        point3d_seq = np.zeros((self.seq_length, self.num_classes, 3))
        
        missing_frames = False
        
        for i in range(0, self.seq_length):
            if frame_num - i < 0:
                missing_frames = True
                break
            
            point2d_seq[-i-1] = self.points2d[index-i] 
            point3d_seq[-i-1] = self.points3d[index-i] 
            
        if missing_frames:
            n_missing_frames = self.seq_length - i

            point2d_seq[0:-i] = np.tile(point2d_seq[-i], (n_missing_frames, 1, 1)) 
            point3d_seq[0:-i] = np.tile(point3d_seq[-i], (n_missing_frames, 1, 1))
        
        point2d_seq = point2d_seq.reshape((self.seq_length * self.num_classes, 2))
        point3d_seq = point3d_seq.reshape((self.seq_length * self.num_classes, 3))

        return point2d_seq, point3d_seq

