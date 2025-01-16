# from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import os

# from annotator.util import HWC3
import random
from tqdm import tqdm

VITON_HD_PATH = "/home/pengjie/data/VITON-HD"
DressCode_PATH = "/home/pengjie/data/DressCode" # add
SHHQ_PATH = "/mnt/lustre/pengjie/data/SHHQ/"

Headwear = (127,255,212)
Hair = (255,0,0)
Glove = (213,140,88)
Eyeglasses = (0,100,0)
Tops = (255,250,250)
Dress = (255, 250, 205)
Coat = (220, 220, 220)
Socks = (160,140,88)
Pants = (211, 211, 211)
Skin = (144, 238, 144)
Scarf = (150, 26, 181)
Skirt = (250, 235,215)
Face = (16, 78,139)
Shoes = (245, 222, 179)
Bag = (255,140,0)
Accessories = (50, 205,50)

TAG = {
    'TopsCoat': 1,
    'Pants': 2,
    'Dress': 3,
    'Skirt': 4,
    'Shoes': 5,
    'Headwear': 6,
    'Eyeglasses': 7,
    'Bag': 8,
    'Scarf': 9,
}

REL_PATH = {
    'TopsCoat': "cloth/topscoat",
    'Pants': "cloth/pants",
    'Dress': "cloth/dress",
    'Skirt': "cloth/skirt",
    'Shoes': "category/shoes",
    'Headwear': "category/headwear",
    'Eyeglasses': "category/eyeglasses",
    'Bag': "category/bag",
    'Scarf': "category/scarf",
}

REP_NUM = {
    'TopsCoat': 1,
    'Pants': 1,
    'Dress': 1,
    'Skirt': 2,
    'Shoes': 1,
    'Headwear': 2,
    'Eyeglasses': 2,
    'Bag': 2,
    'Scarf': 9,
}


def get_image(path, is_flip=False, resize_shape=None):

    image = Image.open(path).convert("RGB")

    if resize_shape is not None:
        image = image.resize(resize_shape)

    image = np.array(image)
    if is_flip:
        image = np.ascontiguousarray(image[:, ::-1, ::])
    image = 2*torch.tensor(image).float() / 255. - 1
    image = image.permute(2, 0, 1) # 锟斤拷H x W x C锟斤拷锟竭讹拷x锟斤拷锟絰通锟斤拷锟斤拷锟斤拷转锟斤拷为C x H x W
    return image

def retain_specific_colors(image, label, colors):
    """
    Retain specific colors in an image based on a label image.

    :param image: PIL Image, the image to process.
    :param label: PIL Image, the label image.
    :param colors: List of tuples, the colors to retain in the format (R, G, B).
    :return: PIL Image, the processed image.
    """
    if image.size != label.size:
        raise ValueError("Image and label must have the same size")

    # 锟斤拷锟斤拷一锟斤拷锟铰的帮拷色图锟斤拷锟斤拷锟斤拷锟斤拷锟?
    output = Image.new("RGB", image.size, "white")
    pixels_image = image.load()
    pixels_label = label.load()
    pixels_output = output.load()

    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if pixels_label[x, y][:3] in colors:  # 锟斤拷锟絣abel图锟斤拷锟叫碉拷锟斤拷色锟角凤拷锟斤拷锟叫憋拷锟斤拷
                pixels_output[x, y] = pixels_image[x, y]  # 锟斤拷锟斤拷原图锟斤拷色
            # 锟斤拷锟津，憋拷锟斤拷锟斤拷色锟斤拷锟斤拷锟斤拷要锟侥憋拷

    return output


class VITON_HD(Dataset):
    def __init__(
        self,
        flip_prob = 0.5,
        is_inference = False,
        clip_tokenizer = None,
        label_num = None
    ):

        if label_num is not None:
            assert is_inference == True
        if is_inference == False:
            assert label_num == None


        print("is_inference:", is_inference)


        self.data_root = VITON_HD_PATH
        
        if not is_inference:
            self.split = "train"
            self.flip_prob = flip_prob
            all_seeds_path = os.path.join(self.data_root, 'train_pairs.txt')
        else:
            self.split = "inference"
            self.flip_prob = 0.0
            all_seeds_path = os.path.join(self.data_root, 'test_pairs.txt')
            # 为训锟斤拷锟斤拷锟斤拷锟斤拷伪锟斤拷签
            if label_num is not None:
                all_seeds_path = os.path.join(self.data_root, 'train_pairs.txt')

        self.seeds = []
        self.seeds_c = []
        self.rep_tag = []
        
        if self.split == "train":

            with open(all_seeds_path, 'r') as f:
                for line in f.readlines():
                    im_name = line.strip().split()[0]
                    self.seeds.append(im_name)
                    self.seeds_c.append(im_name)

        elif self.split == "inference":

            rep_num = 1 # rep_num = label_num if label_num is not None else 1

            seeds = []
            seeds_c = []
            with open(all_seeds_path, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    seeds.append(im_name)
                    seeds_c.append(c_name)
            # seeds = seeds[:3]
            self.seeds = seeds
            self.seeds_c = seeds_c
            """
            for _ in range(rep_num):
                self.seeds.extend(seeds)
            """
            for i in range(rep_num):
                random.seed(i)
                #random.shuffle(seeds)
                #self.seeds_c.extend(seeds)
                self.rep_tag.extend([i]*len(seeds))


        else:
            raise NotImplementedError
        
        vton_caption = os.path.join(self.data_root, 'captions.json') # VITON-HD-MGD锟斤拷锟絚aptions.json

        with open(vton_caption) as f:
            # self.captions_dict = json.load(f)['items']
            self.captions_dict = json.load(f)
            self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        self.clip_tokenizer = clip_tokenizer
        self.label_num = label_num

        # self.seeds = self.seeds[:1000]
        # self.seeds_c = self.seeds_c[:1000]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int):

        seed = self.seeds[i]
        seeds_c = self.seeds_c[i]
        print("seed", seed)
        print("seeds_c", seeds_c)

        directory = "train" if (self.split == "train" or self.label_num is not None) else "test"
        agnostic_path = os.path.join(self.data_root, directory, "agnostic-v3.2", seed)
        person_path = os.path.join(self.data_root, directory, "image", seed)
        densepose_path = os.path.join(self.data_root, directory, "image-densepose-label", seed.split(".")[0] + ".png")
        parse_agnostic_path = os.path.join(self.data_root, directory, "image-parse-agnostic-v3.2", seed.split(".")[0] + ".png")
        cloth_path = os.path.join(self.data_root, directory, "cloth", seeds_c)
        

        prompt = self.captions_dict[seeds_c.split('_')[0]]
        prompt = ", ".join(prompt)

        is_flip = random.random() < self.flip_prob
        
        agnostic = get_image(agnostic_path, is_flip=is_flip)
        person = get_image(person_path, is_flip=is_flip)
        densepose = get_image(densepose_path, is_flip=is_flip)
        parse_agnostic = get_image(parse_agnostic_path, is_flip=is_flip)
        cloth = get_image(cloth_path, is_flip=is_flip)

        prompt_ = "a photo of a" + " " + prompt

        if self.clip_tokenizer is not None:
            # print(222)
            prompt = self.clip_tokenizer([prompt_]).squeeze(0)

        if self.split == "inference":
            return dict(agnostic=agnostic, person=person, densepose=densepose, parse_agnostic=parse_agnostic, cloth=cloth, prompt=prompt, text_prompt=prompt_,seed=seed, dataset="VITON-HD", tag=TAG['TopsCoat'], class_tag='TopsCoat', rep_tag=self.rep_tag[i])
        else:
            return dict(agnostic=agnostic, person=person, densepose=densepose, parse_agnostic=parse_agnostic, cloth=cloth, prompt=prompt, tag=TAG['TopsCoat'], class_tag='TopsCoat')


class DressCode(Dataset): # add
    def __init__(
            self,
            flip_prob=0.5,
            is_inference=False,
            clip_tokenizer=None,
            label_num=None
    ):

        if label_num is not None:
            assert is_inference == True
        if is_inference == False:
            assert label_num == None

        print("is_inference:", is_inference)

        self.data_root = DressCode_PATH

        if not is_inference:
            self.split = "train"
            self.flip_prob = flip_prob
            all_seeds_path = os.path.join(self.data_root, "upper_body", 'train_pairs.txt')
        else:
            self.split = "inference"
            self.flip_prob = 0.0
            all_seeds_path = os.path.join(self.data_root, "upper_body", 'test_pairs_paired.txt') # don't know
            # 为训锟斤拷锟斤拷锟斤拷锟斤拷伪锟斤拷签
            if label_num is not None:
                all_seeds_path = os.path.join(self.data_root, "upper_body", 'train_pairs.txt')

        self.seeds = []
        self.seeds_c = []
        self.rep_tag = []

        if self.split == "train":

            with open(all_seeds_path, 'r') as f:
                for line in f.readlines():
                    im_name = line.strip().split()[0]
                    self.seeds.append(im_name)
                    self.seeds_c.append(im_name)

        elif self.split == "inference":

            rep_num = label_num if label_num is not None else 1

            seeds = []
            with open(all_seeds_path, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    seeds.append(im_name)
            # seeds = seeds[:3]

            for _ in range(rep_num):
                self.seeds.extend(seeds)

            for i in range(rep_num):
                random.seed(i)
                random.shuffle(seeds)
                self.seeds_c.extend(seeds)
                self.rep_tag.extend([i] * len(seeds))

        else:
            raise NotImplementedError

        vton_caption = os.path.join(self.data_root, 'captions.json')  # VITON-HD-MGD锟斤拷锟絚aptions.json

        with open(vton_caption) as f:
            # self.captions_dict = json.load(f)['items']
            self.captions_dict = json.load(f)
            self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        self.clip_tokenizer = clip_tokenizer
        self.label_num = label_num

        # self.seeds = self.seeds[:1000]
        # self.seeds_c = self.seeds_c[:1000]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int):

        seed = self.seeds[i]
        seeds_c = self.seeds_c[i]

        #directory = "train" if (self.split == "train" or self.label_num is not None) else "test"
        directory = "upper_body"
        agnostic_path = os.path.join(self.data_root, directory, "agnostic-v3.2", seed) #####
        person_path = os.path.join(self.data_root, directory, "images", seed) ##
        densepose_path = os.path.join(self.data_root, directory, "dense", seed.split(".")[0][:-1] + "5.png") # "image-densepose-label"
        parse_agnostic_path = os.path.join(self.data_root, directory, "image-parse-agnostic-v3.2",
                                           seed.split(".")[0][:-1] + "4.png") # "image-parse-agnostic-v3.2" #####
        cloth_path = os.path.join(self.data_root, directory, "images", seeds_c) ##

        prompt = self.captions_dict[seeds_c.split('_')[0]]
        prompt = ", ".join(prompt)

        is_flip = random.random() < self.flip_prob

        agnostic = get_image(agnostic_path, is_flip=is_flip)
        person = get_image(person_path, is_flip=is_flip)
        densepose = get_image(densepose_path, is_flip=is_flip)
        parse_agnostic = get_image(parse_agnostic_path, is_flip=is_flip)
        cloth = get_image(cloth_path, is_flip=is_flip)

        prompt_ = "a photo of a" + " " + prompt

        if self.clip_tokenizer is not None:
            # print(222)
            prompt = self.clip_tokenizer([prompt_]).squeeze(0)

        if self.split == "inference":
            return dict(agnostic=agnostic, person=person, densepose=densepose, parse_agnostic=parse_agnostic,
                        cloth=cloth, prompt=prompt, text_prompt=prompt_, seed=seed, dataset="VITON-HD",
                        tag=TAG['TopsCoat'], class_tag='TopsCoat', rep_tag=self.rep_tag[i])
        else:
            return dict(agnostic=agnostic, person=person, densepose=densepose, parse_agnostic=parse_agnostic,
                        cloth=cloth, prompt=prompt, tag=TAG['TopsCoat'], class_tag='TopsCoat')


class SHHQ(Dataset):
    def __init__(
        self,
        flip_prob = 0.5,
        is_inference = False,
        clip_transformer = None,
        label_num = None
    ):
        if label_num is not None:
            assert is_inference == True
        if is_inference == False:
            assert label_num == None

        print("is_inference:", is_inference)
        self.data_root = SHHQ_PATH
        print("self.data_root:",self.data_root)

        if not is_inference:
            self.split = "train"
            self.flip_prob = flip_prob
            all_seeds_path = os.path.join(self.data_root, 'train_pairs.txt')
        else:
            self.split = "inference"
            self.flip_prob = 0.0
            all_seeds_path = os.path.join(self.data_root, 'test_pairs.txt')
            # 为训锟斤拷锟斤拷锟斤拷锟斤拷伪锟斤拷签
            if label_num is not None:
                all_seeds_path = os.path.join(self.data_root, 'train_pairs.txt')

        with open(all_seeds_path, 'r') as file:
            all_seeds = set(line.strip().split(" ")[0].split(".")[0] for line in file)

        
        components = ['TopsCoat','Pants', 'Dress', 'Skirt', 
            'Shoes', 'Headwear', 'Eyeglasses', 'Bag', 'Scarf']

        self.seeds = []
        self.seeds_c = []
        self.class_tag = []
        self.rep_tag = []
        

        for component in components:

            rel_path = REL_PATH[component]
            seeds_path = os.path.join(self.data_root, rel_path.split(os.sep)[0], component + ".txt")
            seeds_ = []
            class_tag_ = []
            with open(seeds_path, 'r') as f:
                for line in f.readlines():
                    im_name = line.strip().split()[0]
                    if im_name.split(".")[0] in all_seeds:
                        seeds_.append(im_name)
                        class_tag_.append(component)

            # seeds_ = seeds_[:3]
            # class_tag_= class_tag_[:3]

            rep_num = REP_NUM[component] if self.split == "train" else 1
            # 锟斤拷锟斤拷伪锟斤拷签锟斤拷锟?
            if label_num is not None:
                rep_num = label_num

            for _ in range(1, 2):
                self.seeds.extend(seeds_)

            for i in range(1, 2):
                if self.split == "inference":
                    random.seed(i)
                    random.shuffle(seeds_)
                self.seeds_c.extend(seeds_)
                self.class_tag.extend(class_tag_)
                self.rep_tag.extend([i]*len(seeds_))
            print("Add component", component, "num", rep_num*len(seeds_))

        self.clip_transformer = clip_transformer

        assert len(self.seeds) == len(self.seeds_c) == len(self.class_tag)

        # self.seeds = self.seeds[:2000]
        # self.seeds_c = self.seeds_c[:2000]


    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int):

        seed = self.seeds[i]
        seeds_c = self.seeds_c[i]
        class_tag = self.class_tag[i]

        person_path = os.path.join(self.data_root, "no_segment", seed)
        print("person_path:", person_path)
        densepose_path = os.path.join(self.data_root, "image-densepose-label", seed.split(".")[0] + ".png")
        
        agnostic_path = os.path.join(self.data_root, REL_PATH[class_tag], "agnostic-v3.3", seed)
        cloth_path = os.path.join(self.data_root, REL_PATH[class_tag], "guided_images", seeds_c)
        print("cloth_path:", cloth_path)
        # parsing_png_path = os.path.join(self.data_root, "parsing_png", seed.split(".")[0] + ".png") 
        parsing_agnostic_path = os.path.join(self.data_root, REL_PATH[class_tag], "parsing-agnostic-v3.3", seed.split(".")[0] + ".png") 

        is_flip = random.random() < self.flip_prob
        
        agnostic = get_image(agnostic_path, is_flip=is_flip)
        person = get_image(person_path, is_flip=is_flip)
        densepose = get_image(densepose_path, is_flip=is_flip)
        parsing_agnostic = get_image(parsing_agnostic_path, is_flip=is_flip)
        cloth_torch = get_image(cloth_path, is_flip=is_flip, resize_shape=(1024, 1024))

        if self.clip_transformer is not None:
            cloth = Image.open(cloth_path)
            # cloth.save("./cloth.jpg")
            if is_flip:
                cloth = cloth.transpose(Image.FLIP_LEFT_RIGHT)
            cloth = self.clip_transformer(cloth)
            # print(111)
        else:
            cloth = get_image(cloth_path)

        if self.split == "inference":
            return dict(agnostic=agnostic, person=person, densepose=densepose, parse_agnostic=parsing_agnostic, cloth=cloth, cloth_torch=cloth_torch, seed=seed, dataset="SHHQ", tag=TAG[class_tag], class_tag=class_tag, rep_tag=self.rep_tag[i])
        else:
            return dict(agnostic=agnostic, person=person, densepose=densepose, parse_agnostic=parsing_agnostic, cloth=cloth, tag=TAG[class_tag], class_tag=class_tag)



def create_dataloader(num_workers, batch_size, is_inference, dataset="VITON_HD", clip_transformer=None, clip_tokenizer=None, label_num=None):

    if dataset == "VITON_HD":
        instance = VITON_HD(is_inference=is_inference, clip_tokenizer=clip_tokenizer, label_num=label_num)
    elif dataset == "SHHQ":
        instance = SHHQ(is_inference=is_inference, clip_transformer=clip_transformer, label_num=label_num)
    else:
        instance = DressCode(is_inference=is_inference, clip_tokenizer=clip_tokenizer, label_num=label_num)

    phase = 'val' if is_inference else 'training'
    print("%s dataset of size %d was created" %
          (phase, len(instance)))
    print("num_workers:", num_workers)
    
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=batch_size,
        # sampler=data_sampler(instance, shuffle=not is_inference, distributed=distributed),
        drop_last=not is_inference,
        # 只锟斤拷锟斤拷锟斤拷伪锟斤拷签锟斤拷锟斤拷要锟斤拷锟斤拷锟斤拷
        # shuffle=not is_inference,
        shuffle=label_num is None,
        # shuffle=True,
        num_workers=num_workers,
    )          

    return dataloader


class MultiDataLoader:
    def __init__(self, *dataloaders, frequencies):
        if len(dataloaders) != len(frequencies):
            raise ValueError("Number of dataloaders and frequencies must match")

        self.dataloaders = dataloaders
        self.frequencies = frequencies

        self.iterators = [iter(dataloader) for dataloader in dataloaders]

        self.total_frequency = sum(frequencies)
        self.counters = [0] * len(dataloaders)

        # Initialize progress bars and counters
        self.pbars = [tqdm(total=len(dataloader), desc=f"Dataset {i + 1} (Epoch 0)") for i, dataloader in
                      enumerate(dataloaders)]
        self.iteration_counts = [0] * len(dataloaders)

    def __iter__(self):
        return self

    # def __next__(self):
    #     for i, (dataloader, freq) in enumerate(zip(self.dataloaders, self.frequencies)):
    #         if self.counters[i] < freq:
    #             try:
    #                 data = next(self.iterators[i])
    #             except StopIteration:
    #                 self.iteration_counts[i] += 1  # Increase iteration count
    #                 self.iterators[i] = iter(dataloader)  # Restart iterator
    #                 self.pbars[i].set_description(f"Dataset {i+1} (Epoch {self.iteration_counts[i]})")  # Update progress bar description with new epoch
    #                 self.pbars[i].reset()  # Reset progress bar

    #             self.pbars[i].update(1)  # Update progress bar
    #             self.counters[i] += 1
    #             return data

    #     self.counters = [0] * len(self.dataloaders)  # Reset counters
    #     return self.__next__()

    def __next__(self):
        while True:
            all_counters_full = True
            for i, (dataloader, freq) in enumerate(zip(self.dataloaders, self.frequencies)):
                if self.counters[i] < freq:
                    all_counters_full = False
                    try:
                        data = next(self.iterators[i])
                    except StopIteration:
                        self.iteration_counts[i] += 1
                        self.iterators[i] = iter(dataloader)
                        self.pbars[i].set_description(f"Dataset {i + 1} (Epoch {self.iteration_counts[i]})")
                        self.pbars[i].reset()
                        data = next(self.iterators[i])  # 鑾峰彇鏂癳poch鐨勭涓€涓暟鎹?

                    self.pbars[i].update(1)
                    self.counters[i] += 1
                    return data

            if all_counters_full:
                # 濡傛灉鎵€鏈夎鏁板櫒閮藉凡婊★紝鍒欓噸缃畠浠?
                self.counters = [0] * len(self.dataloaders)

    def close_progress_bars(self):
        for pbar in self.pbars:
            pbar.close()