from pathlib import Path

import pandas as pd
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from itertools import chain
import asyncio
import aiofiles
from embed_explore import align_images_given_img
from io import BytesIO

exam_record = namedtuple('exam_foo',
                         ['eid', 'label',
                          "logit_yr1", "logit_yr2", "logit_yr3", "logit_yr4", "logit_yr5",
                          "year1_risk", "year2_risk", "year3_risk", "year4_risk", "year5_risk",
                          'l_cc_img', 'l_cc_path', 'r_cc_img', 'r_cc_path', 'l_mlo_img', 'l_mlo_path',
                          'r_mlo_img', 'r_mlo_path'])


class Mammo_CLIPMetadataset(Dataset):
    """
    Creates a torch dataset out of a MIRAI metadata csv.
    Rows are exam_record named tuples.
    """

    def __init__(self, metadata_frame, dataset=None, allow_incomplete=False, verbose=False, transforms=None,
                 mode="training",
                 bu_path="/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/External/BU_Mammo/mammoclip",
                 load_images_async=True, align_images=True, oversample_cancer_rate=None, multiple_pairs_per_exam=False,
                 label_col="cancer1yr_updated", use_mean_risk=False):
        """

        only_complete - both views, both lateralities
        """
        super().__init__()
        assert not (
                align_images and multiple_pairs_per_exam), "Error: Alignment not currently supported with multiple pairs per exam"
        self.exams = []
        self.bu_path = bu_path
        self.transforms = transforms
        self.load_images_async = load_images_async
        self.align_images = align_images
        self.mode = mode
        self.dataset = dataset
        self.mean = 0.3089279
        self.std = 0.25053555408335154
        # Whether we want to average over multiple image pairings for each view
        self.multiple_pairs_per_exam = multiple_pairs_per_exam
        print("Using oversample_cancer_rate of", oversample_cancer_rate)

        # This is necessary to make batching play nice. If we allow different exams to have different
        # numbers of image, we will get jagged input tensors. This is a workaround where we set a max
        # number of images, and pad exams with less than the max number of images with null images.
        # The null images are explicitly handled in the forward pass of the model when present.
        self.max_per_view = 4

        for i, eid in enumerate(metadata_frame['exam_id'].unique()):

            cur_exam = metadata_frame[metadata_frame['exam_id'].values == eid]

            if not self.multiple_pairs_per_exam:
                patient_exam = {'MLO': {'L': None, 'R': None},
                                'CC': {'L': None, 'R': None}}
            else:
                patient_exam = {'MLO': [],
                                'CC': []}
            complete_exam = True
            for view in patient_exam.keys():
                def indices_for_side_view(side):
                    indices = np.logical_and(cur_exam['view'].values == view, cur_exam['laterality'].values == side)
                    return indices

                if len(cur_exam[indices_for_side_view('L')]['file_path'].values) == 0:
                    if verbose:
                        print("Missing {} {} view for exam {}".format(view, laterality, eid))
                    complete_exam = False
                    continue

                elif not self.multiple_pairs_per_exam:
                    for laterality in ['L', 'R']:
                        if len(cur_exam[indices_for_side_view(laterality)]['file_path'].values) == 0:
                            if verbose:
                                print("Missing {} {} view for exam {}".format(view, laterality, eid))
                            complete_exam = False
                            continue
                        old_path = cur_exam[indices_for_side_view(laterality)]['file_path'].values[-1]
                        patient_exam[view][laterality] = old_path

                else:
                    for l_path in cur_exam[indices_for_side_view('L')]['file_path'].values:
                        cur_row = cur_exam[cur_exam['file_path'] == l_path]
                        # This line is checking for nan
                        if type(cur_row['matched_image']) is str:
                            patient_exam[view].append({'L': l_path, 'R': cur_row['matched_image'].values[0]})
                        else:
                            r_imgs = cur_exam[indices_for_side_view('R')]['file_path'].values
                            if len(r_imgs) > 0:
                                patient_exam[view].append(
                                    {'L': l_path, 'R': cur_exam[indices_for_side_view('R')]['file_path'].values[0]})

            if allow_incomplete or complete_exam:
                # exam is cached with just the paths - but the images are loaded from __getitem__
                # print(cur_exam)
                # print(cur_exam[label_col])
                label = 1 if (cur_exam[label_col].values == 1).any() else 0
                label = torch.tensor(label)
                logit_yr1 = torch.tensor(cur_exam["logit_yr1"].values[0])
                logit_yr2 = torch.tensor(cur_exam["logit_yr2"].values[0])
                logit_yr3 = torch.tensor(cur_exam["logit_yr3"].values[0])
                logit_yr4 = torch.tensor(cur_exam["logit_yr4"].values[0])
                logit_yr5 = torch.tensor(cur_exam["logit_yr5_mean"].values[0]) if use_mean_risk else torch.tensor(
                    cur_exam["logit_yr5"].values[0])
                year1_risk = torch.tensor(cur_exam["1_year_risk"].values[0])
                year2_risk = torch.tensor(cur_exam["2_year_risk"].values[0])
                year3_risk = torch.tensor(cur_exam["3_year_risk"].values[0])
                year4_risk = torch.tensor(cur_exam["4_year_risk"].values[0])
                year5_risk = torch.tensor(cur_exam["5_year_risk"].values[0])

                # print(f"logit_yr5: {logit_yr5}")
                # print(f"logit_yr5_mean: {torch.tensor(cur_exam['logit_yr5_mean'].values[0])}")
                # print(f"logit_yr5: {torch.tensor(cur_exam['logit_yr5'].values[0])}")
                # print(xxx)

                def add_record():
                    if self.multiple_pairs_per_exam:
                        self.exams.append((eid, label, patient_exam))
                    else:
                        self.exams.append(
                            exam_record(
                                eid, label,
                                logit_yr1, logit_yr2, logit_yr3, logit_yr4, logit_yr5,
                                year1_risk, year2_risk, year3_risk, year4_risk, year5_risk,
                                patient_exam['CC']['L'],
                                None,
                                patient_exam['CC']['R'],
                                None,
                                patient_exam['MLO']['L'],
                                None,
                                patient_exam['MLO']['R'],
                                None))

                if oversample_cancer_rate is None:
                    add_record()
                elif label == 1:
                    for i in range(oversample_cancer_rate):
                        add_record()
                else:
                    add_record()

        # print(self.dataset)
        # print(self.exams)
        # print(len(self.exams))
        # print(xxx)

    def load_img_path(self, path):
        img_path = None
        if self.dataset.lower() == "rsna":
            img_path = path
        elif self.dataset.lower() == "bu":
            img_path = str(path)
            exam_id = Path(img_path).parent.name
            image_name = Path(img_path).name
            if "controls" in img_path:
                img_path = Path(self.bu_path) / "controls" / "test_images_png" / exam_id / image_name
            elif "cases" in img_path:
                img_path = Path(self.bu_path) / "cases" / "test_images_png" / exam_id / image_name

        return img_path

    def process_img(self, img):
        if self.transforms:
            img = np.array(img)
            augmented = self.transforms(image=img)
            img = augmented['image']
            img = img.astype('float32')
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)
        else:
            img = np.array(img)
            img = img.astype('float32')
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)

        return img.unsqueeze(0)

    def load_img(self, path):
        img_path = self.load_img_path(path)
        img = Image.open(img_path).convert('RGB')
        return self.process_img(img)

    async def load_img_async(self, path):
        """
        Load images asynchronously. The file load is moved to another thread.
        @returns cv2 image from the path
        """
        img_path = self.load_img_path(path)
        async with aiofiles.open(img_path, mode='rb') as file:
            file_contents = await file.read()
        img = Image.open(BytesIO(file_contents)).convert('RGB')
        return self.process_img(img), str(img_path)

    async def load_imgs_async(self, paths):
        """
        Load a batch of images using asyncio in parallel.
        """
        coroutines = [self.load_img_async(path) for path in paths]
        return await asyncio.gather(*coroutines)

    def __len__(self):
        return len(self.exams)

    def __getitem__(self, index):
        exam = self.exams[index]
        # if exam[3] is not None:
        #    return exam
        # else:
        if not self.multiple_pairs_per_exam:
            paths = exam[12::2]
        else:
            patient_exam = exam[2]

            def paths_from_exam(view):
                res = []
                for i in range(len(patient_exam[view])):
                    res = res + [patient_exam[view][i]['L'], patient_exam[view][i]['R']]

                if len(res) < self.max_per_view * 2:
                    return res + [None for j in range(self.max_per_view * 2 - len(res))]
                else:
                    return res[:self.max_per_view * 2]

            paths = paths_from_exam('CC') + paths_from_exam('MLO')

        if self.multiple_pairs_per_exam:
            if self.load_images_async:
                path_image_pairs = asyncio.run(self.load_imgs_async(paths))
            else:
                path_image_pairs = [(self.load_img(path), path) for path in paths]

            res_tuples = []
            cur_view = 'CC'
            for i in range(0, len(path_image_pairs), 2):
                if i >= len(paths_from_exam('CC')):
                    cur_view = 'MLO'
                a = path_image_pairs[i][0]
                b = path_image_pairs[i + 1][0]
                res_tuples = res_tuples + [
                    (a, b, cur_view)]  # [(path_image_pairs[i][0], path_image_pairs[i+1][0], cur_view)]

            return ([exam[0], exam[1], res_tuples])

        else:
            if self.load_images_async:
                path_image_pairs = asyncio.run(self.load_imgs_async(paths))
            else:
                path_image_pairs = list(map(lambda path: (self.load_img(path), path), paths))

            if self.align_images:
                l_cc, r_cc = align_images_given_img(path_image_pairs[0][0].numpy()[0],
                                                    path_image_pairs[1][0].numpy()[0])
                l_mlo, r_mlo = align_images_given_img(path_image_pairs[2][0].numpy()[0],
                                                      path_image_pairs[3][0].numpy()[0])
                path_image_pairs = [
                    (torch.tensor(l_cc).expand(3, *l_cc.shape).type(torch.FloatTensor), path_image_pairs[0][1]),
                    (torch.tensor(r_cc).expand(3, *r_cc.shape).type(torch.FloatTensor), path_image_pairs[1][1]),
                    (torch.tensor(l_mlo).expand(3, *l_mlo.shape).type(torch.FloatTensor), path_image_pairs[2][1]),
                    (torch.tensor(r_mlo).expand(3, *r_mlo.shape).type(torch.FloatTensor), path_image_pairs[3][1])]

            # self.exams[index] = ([exam[0], exam[1]] + list(chain(*path_image_pairs)))
            return ([
                        exam[0], exam[1],
                        exam[2], exam[3], exam[4], exam[5], exam[6],
                        exam[7], exam[8], exam[9], exam[10], exam[11]
                    ] + list(chain(*path_image_pairs)))
            # if self.dataset == "bu":
            #     return ([
            #                 exam[0], exam[1],
            #                 exam[2], exam[3], exam[4], exam[5], exam[6],
            #                 exam[7], exam[8], exam[9], exam[10], exam[11]
            #             ] + list(chain(*path_image_pairs)))
            # elif self.dataset == "rsna":
            #     return ([exam[0], exam[1]] + list(chain(*path_image_pairs)))
