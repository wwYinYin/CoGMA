import numpy as np

from torch.utils.data import Dataset
import os
from feeders import tools
import torch
import time

bone_pairs = (
    (0, 17), (1, 0), (2, 0), (3, 1), (4, 2), (5, 17),(6, 17), (7, 5), (8, 6), (9, 7), (10, 8), (11, 19),
    (12, 19), (13, 11), (14, 12), (15, 13), (16, 14), (17, 18),(18, 19)
)

class Feeder(Dataset):
    def __init__(self, data_path, method=None,text_embedding=None,random_rot=False, bone=False):
        self.data_pathes=[]
        self.label=[]
        self.infant_id=[]
        self.video_feature=[]
        self.text_feature=[]
        self.text_embedding = text_embedding
        video_embed_path='/home/yinw/data/data_4training_use_video_feature'
        text_embed_path='/home/yinw/data/data_4training_use_text_feature'
        for filename in os.listdir(data_path):
            # carmera_class = int(filename[filename.find('C') + 1:filename.find('C') + 2])
            # if carmera_class != 1:
            #     continue
            action_class = int(filename[filename.find('N') + 1:filename.find('N') + 2])
            self.data_pathes.append(os.path.join(data_path, filename))
            self.label.append(action_class)
            self.infant_id.append(filename[0:4])
            self.video_feature.append(os.path.join(video_embed_path, filename))
            self.text_feature.append(os.path.join(text_embed_path, filename))

        # self.label = np.array(self.label)
        self.random_rot = random_rot
        self.bone = bone
        self.method = method
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        data_path = self.data_pathes[index]
        data_numpy = np.load(data_path).astype(np.float32)
        data_numpy = tools.frame_translation(data_numpy,dimension="3D")
        video_feature=np.load(self.video_feature[index]).astype(np.float32) #(8, 144, 3072)
        text_feature=np.load(self.text_feature[index]).astype(np.float32) #(8, 3072)
        if self.method == 'MAM':
            if self.bone:
                bone_data = np.zeros_like(data_numpy)
                for v1, v2 in bone_pairs:
                    bone_data[:, v1] = data_numpy[:, v1] - data_numpy[:, v2]
                data_numpy=bone_data
            data_numpy = tools.split_data(data_numpy)
            # data_numpy = tools.split_data_8(data_numpy)
            # data_numpy = tools.padding_data(data_numpy)
            data_numpy = np.transpose(data_numpy, (0, 1, 3, 2)) #[16, 900, 3, 20]
            expanded_arr = data_numpy[..., np.newaxis] # [16, 900, 3, 20, 1]
            # 计算差值
            data = expanded_arr - expanded_arr.transpose(0, 1, 2, 4, 3) # [16, 900, 3, 20, 20]
            data=data.transpose(0, 3, 4, 2, 1) # [16, 20, 20, 3, 900]
            data = torch.tensor(data)      
        elif self.bone:
            # start_time = time.time()
            data = tools.calculate_features(data_numpy) #[8, 1200, 20, 3]
            # end_time = time.time()
            # print(f"Execution time: {end_time - start_time} seconds")
            data = torch.tensor(data)

        else:
            if self.random_rot:
                data_numpy = tools.random_rot(data_numpy)
            # if self.bone:
            #     bone_data = np.zeros_like(data_numpy)
            #     for v1, v2 in bone_pairs:
            #         bone_data[:, v1] = data_numpy[:, v1] - data_numpy[:, v2]
            #     data_numpy=bone_data
            # data = tools.split_data(data_numpy)
            data = tools.split_data_8(data_numpy)
            # data = tools.split_data20s(data_numpy)
            data=torch.tensor(data) #[16, 900, 20, 3]
        video_feature=torch.tensor(video_feature)
        text_feature=torch.tensor(text_feature)
        if self.text_embedding is not None:
            # info_embedding = torch.tensor(self.text_embedding[self.infant_id[index]], dtype=torch.float32) #[4,512] 
            info_embedding = self.text_embedding[self.infant_id[index]].clone().detach()   
        else:
            info_embedding = None 
        label = torch.tensor(self.label[index], dtype=torch.long)
        return data, label, info_embedding,video_feature,text_feature


    def top_k(self, score, top_k=1):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)