import torch
import os
from tqdm import tqdm
import shutil
from pathlib import Path

class ImagesManager():
    def __init__(self, dataloader, model, device):
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.model_to_device(self.device)
        
    def model_to_device(self,device):
        self.model.to(device)

    def vecs_to_tensor(self):
        vec_tensor = torch.tensor([[]])
        vec = self.vec_data[0][0]
        vec = torch.unsqueeze(vec, 0)
        vec_tensor = torch.cat((vec_tensor, vec),1)
        for vec in tqdm(self.vec_data[1:]):
            vec = vec[0]
            vec = torch.unsqueeze(vec, 0)
            vec_tensor = torch.cat((vec_tensor, vec),0)
        self.vec_tensor = vec_tensor

    def images_to_vec_data(self):
        dataloader = iter(self.dataloader)
        vec_data = []
        for images, _, img_paths in tqdm(dataloader):

            images = images.to(self.device)
            vecs = self.model(images)
            images.detach().cpu() # remove that ?
            vecs = vecs.detach().cpu()
            for feature_vec, img_path in zip(vecs, img_paths):
                vec_data.append((feature_vec, img_path))
        self.vec_data = vec_data

    def order_images_by_similarity(self, img_idx, search_from_idx=0):
        results = []
        query_tensor = self.vec_tensor[img_idx]
        repeated_query_tensor = query_tensor.repeat(len(self.vec_tensor)-search_from_idx,1)
        limited_searched_tensors = self.vec_tensor[search_from_idx:]
        cos_similaries = torch.nn.functional.cosine_similarity(repeated_query_tensor, limited_searched_tensors)
        for idx_i, cos_similarity in enumerate(cos_similaries):
            img_path = self.vec_data[idx_i][1]
            results.append((idx_i, cos_similarity, img_path))
        results = sorted(results, key=lambda row: row[1], reverse=True)
        return results
    
    def count_cross_similarity(self,):
        cross_similarity = []
        for i in tqdm(range(0,len(self.vec_data))):
            similar_images = self.order_images_by_similarity(i, search_from_idx = 0)
            cross_similarity.append(similar_images)
        self.cross_similarity = cross_similarity

    def seperate_non_uniq(self, target_dir, treshold = 0.85):
        uniq_dir = os.path.join(target_dir,"uniq")
        similar_groups_counter = 0
        used_similar_groups = []
        for query_img_idx, similar_images in tqdm(enumerate(self.cross_similarity)):
            if len(similar_images) == 1 or similar_images[1][1] < treshold:
                # print(similar_images)
                img_path = similar_images[0][2]
                img_name = os.path.basename(img_path)
                Path(uniq_dir).mkdir(parents=True, exist_ok=True)
                new_img_path = os.path.join(uniq_dir,img_name)
                shutil.copyfile(img_path, new_img_path)
            elif query_img_idx not in used_similar_groups:
                non_uniq_dir = os.path.join(target_dir, "similar" + str(similar_groups_counter))
                Path(non_uniq_dir).mkdir(parents=True, exist_ok=True)
                for similar_sample in similar_images:
                    if similar_sample[1] >= treshold:
                        similar_img_path = similar_sample[2]
                        similar_img_idx = similar_sample[0]
                        img_name = os.path.basename(similar_img_path)
                        similar_img_new_path = os.path.join(non_uniq_dir,img_name)
                        shutil.copyfile(similar_img_path, similar_img_new_path)
                        used_similar_groups.append(similar_img_idx)
                    else:
                        break
                similar_groups_counter += 1