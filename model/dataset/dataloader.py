# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the class for CurveLane dataset."""

import os
import json
import cv2
import numpy as np
import warnings
import yaml

from torch.utils.data import Dataset
import torch.utils.data.dataloader
from dataset.utility import get_img_whc, imread, create_subset, resize_by_wh, bgr2rgb, imagenet_normalize, load_json

# 车道线encoder / decoder
from head_lane.lane_codec_utils import trans_to_lane_with_type,delete_repeat_y,delete_nearby_point
from head_lane.lane_spline_interp import spline_interp
from scipy import interpolate
#---------------------------------------------------#
#  数据增强函数
#---------------------------------------------------#
import imgaug as ia
import imgaug.augmenters as iaa

# 车道线
from imgaug.augmentables.lines import LineStringsOnImage
from imgaug.augmentables.lines import LineString as ia_LineString

# 目标检测
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.bbs import BoundingBox as ia_Bbox

#  语义分割增强函数
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def _lane_argue(*,
                image,
                lane_label=None,
                det_label=None,
                seg_label=None,
                do_flip=False,
                do_split=False,
                split_ratio=None,
                ):

    #---------------------------------------------------#
    #  定义增强序列
    #---------------------------------------------------#
    color_shift = iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.5, 1.5)),
        iaa.LinearContrast((1.5, 1.5), per_channel=False),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(0, iaa.Multiply((0.7, 1.3)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(1, iaa.Multiply((0.1, 2)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(2, iaa.Multiply((0.5, 1.5)))),
    ])

    geometry_trans_list = [
            iaa.Fliplr(),
            iaa.TranslateX(px=(-16, 16)),
            iaa.ShearX(shear=(-15, 15)),
            iaa.Rotate(rotate=(-15, 15))
        ]

    if do_flip:
        geometry_trans_list.append(iaa.Flipud())

    if do_split:
        # top right down left
        split_one = iaa.Crop(percent=([0, 0.2], [1 - split_ratio], [0, 0], [0, 0.15]), keep_size=True)# 右边是1-ratio
        split_two = iaa.Crop(percent=([0, 0.2], [0, 0.15], [0, 0], [split_ratio]), keep_size=True)
        split_shift = iaa.OneOf([split_one, split_two])

    else:
        geometry_trans_list.append(iaa.Crop(percent=([0, 0.2], [0, 0.15], [0, 0], [0, 0.15]), keep_size=True))
        split_shift = None

    posion_shift = iaa.SomeOf(4, geometry_trans_list)

    if do_split:
        aug = iaa.Sequential([
            iaa.Sometimes(p=0.6, then_list=color_shift),
            iaa.Sometimes(p=0.6, then_list=split_shift), # 以0.5概率去split debug时候1.0
            iaa.Sometimes(p=0.6, then_list=posion_shift)
        ], random_order=True)

    else:
        aug = iaa.Sequential([
            iaa.Sometimes(p=0.6, then_list=color_shift),
            iaa.Sometimes(p=0.6, then_list=posion_shift)], random_order=True)

    # =========================================
    # 开始数据增强
    # =========================================
    args = dict(images=[image])
    if lane_label is not None:
        lines_tuple = [[(float(pt['x']), float(pt['y'])) for pt in line_spec] for line_spec in lane_label['Lines']]
        lss = [ia_LineString(line_tuple_spec) for line_tuple_spec in lines_tuple]
        lsoi = LineStringsOnImage(lss, shape=image.shape)
        args.update({"line_strings":[lsoi]})

    # =========================================
    # 做语义分割增强
    # =========================================
    if seg_label is not None:
        segmap = SegmentationMapsOnImage(seg_label , shape=image.shape)
        args.update({"segmentation_maps":[segmap]})

    # =========================================
    # 做目标检测增强
    # =========================================
    if det_label is not None:
        bbox_list = [ia_Bbox( *list(one_det_poly[:5])) for one_det_poly in det_label]
        deoi = BoundingBoxesOnImage(bbox_list, shape=image.shape)
        args.update({"bounding_boxes": [deoi]})

    # =========================================
    # 开始增强
    # =========================================
    batch = ia.Batch(**args)
    batch_aug = list(aug.augment_batches([batch]))[0]  # augment_batches returns a generator
    image_aug = batch_aug.images_aug[0]

    # 增强line
    aug_result = dict(images=image_aug)
    if lane_label is not None:
        # lsoi_aug = batch_aug.line_strings_aug[0].clip_out_of_image() # 这个很重要
        lsoi_aug = batch_aug.line_strings_aug[0] # 这里不clip_out_of_image()

        lane_aug = [[dict(x= float(int(kpt.x)), y=float(int(kpt.y))) for kpt in shapely_line.to_keypoints()]
                    for shapely_line in lsoi_aug]

        aug_result.update({"lane_aug": dict(Lines=lane_aug,Labels=None)})

    # 增强detection
    if det_label is not None:
        # 这里clip out of image 会有问题，所以不clip
        deoi_aug = batch_aug.bounding_boxes_aug[0].clip_out_of_image()
        if len(deoi_aug) ==0:
            det_label_aug = np.zeros((0, 5)) # clip后没有目标的情况
        else:
            det_label_aug = np.vstack([np.hstack([det_bbox.coords.reshape(1,4),det_bbox.label.reshape(1,1)]) for det_bbox in deoi_aug])
        aug_result.update({"det_aug":det_label_aug})

    # 增强分割掩码
    if seg_label is not None:
        segmap_aug = batch_aug.segmentation_maps_aug[0]
        aug_result.update({"seg_aug":segmap_aug})

    return aug_result

def _read_curvelane_type_annot(annot_path):
    return load_json(annot_path)

class MultitaskData(Dataset):
    def __init__(self, cfgs, mode):
        """Construct the dataset."""
        super().__init__()
        self.cfgs = cfgs
        self.mode = mode

        # 加载通用
        self.network_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.network_input_height = self.cfgs["dataloader"]["network_input_height"]

        # 数据增强配置
        self.with_aug = self.cfgs["dataloader"]["with_aug"]
        if self.mode == "val": self.with_aug = False

        self.do_split_img = self.cfgs["dataloader"]["do_split"]
        self.do_flip_img = self.cfgs["dataloader"]["do_flip"]

        # 加载检测相关
        self.train_detect = self.cfgs["train"]["train_detect"]
        self.det_class_list = self.cfgs["detection"]["class_list"]
        self.det_num_classes = len(self.det_class_list)
        self.det_class_to_ind = dict(zip(self.det_class_list, range(self.det_num_classes)))

        # 加载分割相关
        self.train_seg = self.cfgs["train"]["train_seg"]
        self.seg_class_list = self.cfgs["segment"]["class_list"]

        # 加载车道线相关
        self.train_lane = self.cfgs["train"]["train_lane"]
        self.k1 = cfgs["lane"]["k1"]
        self.thresh = cfgs["lane"]["thresh"]
        self.threshold_instance = cfgs["lane"]["threshold_instance"]
        self.resize_ratio = cfgs["lane"]["resize_ratio"]
        self.grid_x = int(self.network_input_width / self.resize_ratio)
        self.grid_y = int(self.network_input_height /self.resize_ratio)
        self.ratio_start = cfgs["lane"]["ratio_start"]
        self.ratio_interval = cfgs["lane"]["ratio_interval"]
        self.interpolation = self.cfgs["lane"]["interpolate"]  # 是否插值 如果插值 即为沿长到版边 和原代码就完全一致
        self.points_per_line = self.grid_y

        if not (self.train_lane or self.train_seg or self.train_detect):
            print("must train at least one header")
            exit()


        # 准备数据集
        self.data_list = self.cfgs["dataloader"]["data_list"]
        self.data_list_train = os.path.join(self.data_list, "train.txt")
        self.data_list_valid = os.path.join(self.data_list, "valid.txt")

        dataset_pairs = dict(
            train=create_subset(self.data_list_train,with_lane=self.train_lane,with_seg=self.train_seg,with_detect=self.train_detect),
            val=create_subset(self.data_list_valid,with_lane=self.train_lane,with_seg=self.train_seg,with_detect=self.train_detect)
        )

        if self.mode not in dataset_pairs.keys():
            raise NotImplementedError(f'mode should be one of {dataset_pairs.keys()}')
        self.image_annot_path_pairs = dataset_pairs.get(self.mode)

        # 收集函数
        self.collate_fn = Collater(target_height=self.network_input_height,
                                   target_width=self.network_input_width,
                                   is_lane = self.train_lane,
                                   is_seg=self.train_seg,
                                   is_det=self.train_detect)

    def __len__(self):
        """Get the length.

        :return: the length of the returned iterators.
        :rtype: int
        """
        return len(self.image_annot_path_pairs)

    def __getitem__(self, idx):
        """Get an item of the dataset according to the index.

        :param idx: index
        :type idx: int
        :return: an item of the dataset according to the index
        :rtype: dict
        """
        return self.prepare_img(idx)

    def prepare_img(self, idx):
        """Prepare an image for training.

        :param idx:index
        :type idx: int
        :return: an item of data according to the index
        :rtype: dict
        """
        target_pair = self.image_annot_path_pairs[idx]
        image_arr = imread(target_pair['image_path'])
        whc = get_img_whc(image_arr)

        #---------------------------------------------------#
        #  加入车道线标签
        #---------------------------------------------------#
        if self.train_lane:
            lane_label = self.parse_own_label(load_json(target_pair['annot_path_lane']))
            annot_lane_path = target_pair['annot_path_lane']
        else:
            lane_label = None
            annot_lane_path = None

        # ---------------------------------------------------#
        #  加入语义分割
        # ---------------------------------------------------#
        if self.train_seg:
            seg_label = cv2.imread(target_pair['annot_path_seg'], cv2.IMREAD_UNCHANGED)
        else:
            seg_label = None

        # ---------------------------------------------------#
        #  加入目标检测
        # ---------------------------------------------------#
        if self.train_detect:
            obj_label = self.load_detect_annot(target_pair['annot_path_detect'])
        else:
            obj_label = None

        if DEBUG:
            self.draw_label_on_image(image_arr, lane_label, obj_label, seg_label, "img_org.png")

        # =========================================
        # 数据增强
        # =========================================
        if self.with_aug:
            if self.do_split_img:
                do_split, split_ratio = self.cal_split(image_arr, lane_label)
            else:
                do_split = False
                split_ratio = None

            aug_dict = _lane_argue(image=image_arr,
                                   lane_label=lane_label,
                                   det_label=obj_label,
                                   seg_label=seg_label,
                                   do_flip=self.do_flip_img,
                                   do_split=do_split,
                                   split_ratio=split_ratio
                                   )

            # =========================================
            # 取出数据
            # =========================================
            # 1.图像
            image_arr = aug_dict["images"]

            # 2.车道线
            if self.train_lane:
                lane_label = aug_dict["lane_aug"]

            # 3.覆盖分割
            if self.train_seg:
                seg_label = aug_dict["seg_aug"].arr[:, :, 0].astype(np.uint8)  # 分割标签 原图尺度

            # 4.检测标签
            if self.train_detect:
                obj_label = aug_dict["det_aug"]

            if DEBUG:
                self.draw_label_on_image(image_arr, lane_label, obj_label, seg_label, "img_aug.png")

        # =========================================
        # lane进一步进行decode
        # =========================================
        if self.train_lane:
            x_lists, y_lists = self.cvt_dict_to_list_lane(lane_label)
            data_dict = self.cvt_lane_to_tusimple_format(x_lists, y_lists, whc,target_pair['image_path'])

            # 准备lane gt
            target_lanes, target_h = self.cvt_dense_lane(x_lists, y_lists, whc)

            ground_truth_point, ground_truth_point_binary = self.make_ground_truth_point(target_lanes, target_h)
            ground_truth_instance = self.make_ground_truth_instance(target_lanes, target_h)
        else:
            data_dict = None
            ground_truth_point = None
            ground_truth_instance = None

        # =========================================
        # object进一步进行decode
        # =========================================
        if self.train_detect:
            gt_bboxes, gt_labels = self.transfer_det_center(obj_label,org_shape=(whc["width"],whc["height"]),
                                                            target_shape=(self.network_input_width,self.network_input_height))
        else:
            gt_bboxes=None
            gt_labels=None

        # 图像resize
        network_input_image = bgr2rgb(resize_by_wh(img=image_arr,
                                                   width= self.network_input_width,
                                                   height=self.network_input_height))

        net_input_image_shape = json.dumps(dict(width=self.network_input_width, height=self.network_input_height, channel=3))


        # image meta data
        img_meta = dict()
        img_meta["filename"] = target_pair['image_path']
        img_meta["ori_shape"] = (whc["height"], whc["width"], 3)
        img_meta["img_shape"] = (whc["height"], whc["width"], 3)
        img_meta["pad_shape"] = (whc["height"], whc["width"], 3)
        img_meta["scale_factor"] = [1., 1., 1., 1.]
        img_meta["flip"] = False
        img_meta["flip_direction"] = None
        img_meta["img_norm_cfg"] = {'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                                    'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                                    'to_rgb': True}

        img_meta["border"] = [0, whc["height"], 0, whc["width"]]
        img_meta["batch_input_shape"] = (whc["height"], whc["width"])
        img_metas = [img_meta]


        result = dict(
                      image=np.transpose(imagenet_normalize(img=network_input_image), (2, 0, 1)).astype('float32'),
                      src_image_shape=whc,
                      net_input_image_shape=net_input_image_shape,
                      src_image_path=target_pair['image_path'],
                      annot_lane=json.dumps(lane_label),
                      annot_lane_path=annot_lane_path,
                      gt_lane_points_map=ground_truth_point,
                      gt_lane_points_instance=ground_truth_instance,
                      gt_lane_org_label=data_dict,
                      gt_seg=seg_label,
                      gt_det_box=gt_bboxes,
                      gt_det_label=gt_labels,
                      meta_data=img_metas,
                      )

        return result

    @staticmethod
    def cvt_dict_to_list_lane(lane_labels):
        x_lists = list()
        y_lists = list()
        for lane in lane_labels["Lines"]:
            x_list = list()
            y_list = list()
            for item in lane:
                x_list.append(int(item["x"]))
                y_list.append(int(item["y"]))

            x_lists.append(x_list)
            y_lists.append(y_list)
        return x_lists, y_lists


    def uniform_sample_lane_y_axis(self, x_pt_list, y_pt_list):
        """Ensure y from bottom of image."""
        if len(x_pt_list) < 2 or len(y_pt_list) < 2:
            return -1, -1, [], []

        if self.interpolation:
            max_y = y_pt_list[-1]
            if max_y < self.network_input_height - 1:
                y1 = y_pt_list[-2]
                y2 = y_pt_list[-1]
                x1 = x_pt_list[-2]
                x2 = x_pt_list[-1]

                while max_y < self.network_input_height - 1:
                    y_new = max_y + self.resize_ratio # TODO 这里注意了！！
                    x_new = x1 + (x2 - x1) * (y_new - y1) / (y2 - y1)
                    x_pt_list.append(x_new)
                    y_pt_list.append(y_new)
                    max_y = y_new

        x_list = np.array(x_pt_list)
        y_list = np.array(y_pt_list)  # y from small to big
        if y_list.max() - y_list.min() < 5:  # filter < 5 pixel lane
            return -1, -1, [], []
        if len(y_list) < 4:
            tck = interpolate.splrep(y_list, x_list, k=1, s=0)
        else:
            tck = interpolate.splrep(y_list, x_list, k=3, s=0)

        if self.interpolation:
            startpos = 0
        else:
            startpos = int( (self.network_input_height - 1 -  y_list[-1]) / self.resize_ratio + 1)  # TODO 这里要加1

        endpos = int((self.network_input_height - 1 - y_list[0]) / self.resize_ratio)
        if endpos > self.points_per_line - 1:
            endpos = self.points_per_line - 1
        if startpos >= endpos:
            return -1, -1, [], []

        y_list = []
        expand_pos = endpos
        for i in range(startpos, expand_pos + 1):
            y_list.append(self.network_input_height - 1 - i * self.resize_ratio)
        xlist = interpolate.splev(y_list, tck, der=0)

        for i in range(len(xlist)):
            if xlist[i] == 0:
                xlist[i] += 0.01

        return startpos, endpos, xlist, y_list

    def cvt_dense_lane(self, x_lists, y_lists, whc):
        target_lanes = []
        target_h = []
        temp_lanes = []
        temp_h = []
        gt_lanes_list = []
        ratio_h = self.network_input_height / whc["height"]
        ratio_w = self.network_input_width / whc["width"]

        for l,h in zip(x_lists, y_lists):
            l = np.array(l).reshape(1,-1) * ratio_w
            l = np.clip(l,a_min=0,a_max=self.network_input_width-1)
            h = np.array(h).reshape(1,-1) * ratio_h
            h = np.clip(h,a_min=0,a_max=self.network_input_height-1)
            # l, h = self.make_dense_x(l, h, whc["height"], whc["width"])
            gt_lanes_list.append(np.vstack([l,h]).transpose())

        lane_set = trans_to_lane_with_type(gt_lanes_list)
        for idx,lane in enumerate(lane_set):
            cur_line = lane.lane
            new_lane = delete_repeat_y(cur_line)
            if len(new_lane) < 2:
                x_list = []
                y_list = []
            else:
                interp_lane = spline_interp(lane=new_lane, step_t=1)
                x_list, y_list = delete_nearby_point(interp_lane,self.network_input_width, self.network_input_height)
                # x_pt_list = x_pt_list[::-1]
                # y_pt_list = y_pt_list[::-1]  # y from small to big
                # _, _, x_list, y_list = self.uniform_sample_lane_y_axis(x_pt_list, y_pt_list)


            temp_h.append( np.array(y_list) )
            temp_lanes.append( np.array(x_list) )
        target_lanes.append(np.array(temp_lanes))
        target_h.append(np.array(temp_h))

        return target_lanes, target_h

    @staticmethod
    def make_dense_x(l, h, org_height, org_width):
        out_x = []
        out_y = []

        p_x = -1
        p_y = -1
        for x, y in zip(l, h):
            if x >= 0:
                if p_x < 0:
                    p_x = x
                    p_y = y
                else:
                    if 0 <= x < org_width and 0 <= y < org_height:
                        out_x.append(x)
                        out_y.append(y)

                    for dense_x in range(min(p_x, x), max(p_x, x), 10):
                        dense_y = p_y + abs(p_x - dense_x) * abs(p_y-y)/float(abs(p_x - x))
                        if 0 <= dense_x < org_width and 0 <= dense_y < org_height:
                            out_x.append(dense_x)
                            out_y.append( dense_y )

                    p_x = x
                    p_y = y

        return np.array(out_x), np.array(out_y)

    def get_discrete_pts_poly_fit(self,org_pts, org_height):
        x_list = []
        y_list = []
        for pt in org_pts:
            pt_x = pt[0]
            pt_y = pt[1]
            x_list.append(pt_x)
            y_list.append(pt_y)

        if len(x_list) > 2:
            coeff = np.polyfit(y_list, x_list, 2)
        else:
            coeff = np.polyfit(y_list, x_list, 1)
        func_x_of_y = np.poly1d(coeff)

        h_sample, interval = self.get_hsample(org_height)

        x_inter = func_x_of_y(h_sample)
        x_array = np.array(x_list)
        x_min = np.min(x_array)
        x_max = np.max(x_array)

        y_array = np.array(y_list)
        y_min = np.min(y_array)

        x_negnect = np.logical_or((x_inter < x_min), (x_inter > x_max))
        x_negnect_ = np.logical_or((h_sample > org_height), (h_sample < y_min - 2 * interval))
        x_inter[x_negnect] = -2
        x_inter[x_negnect_] = -2
        return x_inter

    def get_hsample(self,img_height):
        start_y = int(img_height / self.ratio_start)
        end_y = img_height
        interval = int(img_height / self.ratio_interval)
        h_sample = np.arange(start_y, end_y, interval).astype(np.float)
        return h_sample, interval

    # 分段线性插值
    def get_discrete_pts_line_fit(self, x_list, y_list, org_width, org_height):
        x_array = np.array(x_list)
        y_array = np.array(y_list)
        y_min = np.min(y_array)
        y_max = np.max(y_array)

        h_sample, interval = self.get_hsample(org_height)
        x_inter = np.zeros(h_sample.shape[0])
        for idx, tmp_y in enumerate(h_sample):
            if tmp_y < y_min and tmp_y + interval < y_min:
                x_inter[idx] = -2
                continue
            if tmp_y < y_min <= tmp_y + interval:
                # 找前两个点
                index_up = -1
                index_down = -2

            elif y_min <= tmp_y <= y_max:
                # 找上下两个点
                index_up = np.where((y_array - tmp_y) <= 0)[0][0]
                index_down = np.where((y_array - tmp_y) >= 0)[0][-1]

            else:
                # 找最后两个点
                index_up = 1
                index_down = 0

            x1, y1 = x_array[index_down], y_array[index_down]
            x2, y2 = x_array[index_up], y_array[index_up]

            # 线性插值
            if index_down == index_up:
                tmp_val = x1
            else:
                tmp_val = x1 + ((tmp_y - y1) / (y2 - y1 + 1e-9)) * (x2 - x1)

            if tmp_val < 0 or tmp_val > org_width:
                x_inter[idx] = -2
            else:
                x_inter[idx] = tmp_val
        return x_inter


    def cvt_lane_to_tusimple_format(self, x_lists, y_lists, whc, img_path):
        data_lane_dict = {}
        lane_lists = []
        org_width = whc["width"]
        org_height = whc["height"]
        h_sample = list(self.get_hsample(org_height)[0])

        for (x_list_, y_list_) in zip(x_lists, y_lists):
            x_inter = self.get_discrete_pts_line_fit(x_list_, y_list_, org_width, org_height)
            lane_lists.append(list(x_inter))

        data_lane_dict["lanes"] = lane_lists
        data_lane_dict["h_samples"] = h_sample
        data_lane_dict["raw_file"] = img_path
        return data_lane_dict

    @staticmethod
    def sort_batch_along_y(target_lanes, target_h):
        out_x = []
        out_y = []

        for x_batch, y_batch in zip(target_lanes, target_h):
            temp_x = []
            temp_y = []
            for x, y, in zip(x_batch, y_batch):
                ind = np.argsort(y, axis=0)
                sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
                sorted_y = np.take_along_axis(y, ind[::-1], axis=0)
                temp_x.append(sorted_x)
                temp_y.append(sorted_y)
            out_x.append(temp_x)
            out_y.append(temp_y)

        return out_x, out_y


    def make_ground_truth_point(self, target_lanes, target_h):

        target_lanes, target_h = self.sort_batch_along_y(target_lanes, target_h)

        ground = np.zeros((len(target_lanes), 3, self.grid_y, self.grid_x))
        ground_binary = np.zeros((len(target_lanes), 1, self.grid_y, self.grid_x))

        for batch_index, batch in enumerate(target_lanes):
            for lane_index, lane in enumerate(batch):
                for point_index, point in enumerate(lane):
                    if point > 0:
                        x_index = int(point/self.resize_ratio)
                        y_index = int(target_h[batch_index][lane_index][point_index]/self.resize_ratio)
                        ground[batch_index][0][y_index][x_index] = 1.0
                        ground[batch_index][1][y_index][x_index]= (point*1.0/self.resize_ratio) - x_index
                        ground[batch_index][2][y_index][x_index] = (target_h[batch_index][lane_index][point_index]*1.0/self.resize_ratio) - y_index
                        ground_binary[batch_index][0][y_index][x_index] = 1

        return ground, ground_binary

    #####################################################
    ## Make ground truth for instance feature
    #####################################################
    def make_ground_truth_instance(self, target_lanes, target_h):

        ground = np.zeros((len(target_lanes), 1, self.grid_y*self.grid_x, self.grid_y*self.grid_x))

        for batch_index, batch in enumerate(target_lanes):
            temp = np.zeros((1, self.grid_y, self.grid_x))
            lane_cluster = 1
            for lane_index, lane in enumerate(batch):
                previous_x_index = 0
                previous_y_index = 0
                for point_index, point in enumerate(lane):
                    if point > 0:
                        x_index = int(point/self.resize_ratio)
                        y_index = int(target_h[batch_index][lane_index][point_index]/self.resize_ratio)
                        temp[0][y_index][x_index] = lane_cluster
                    if previous_x_index != 0 or previous_y_index != 0: #interpolation make more dense data
                        temp_x = previous_x_index
                        temp_y = previous_y_index
                        while False:
                            delta_x = 0
                            delta_y = 0
                            temp[0][temp_y][temp_x] = lane_cluster
                            if temp_x < x_index:
                                temp[0][temp_y][temp_x+1] = lane_cluster
                                delta_x = 1
                            elif temp_x > x_index:
                                temp[0][temp_y][temp_x-1] = lane_cluster
                                delta_x = -1
                            if temp_y < y_index:
                                temp[0][temp_y+1][temp_x] = lane_cluster
                                delta_y = 1
                            elif temp_y > y_index:
                                temp[0][temp_y-1][temp_x] = lane_cluster
                                delta_y = -1
                            temp_x += delta_x
                            temp_y += delta_y
                            if temp_x == x_index and temp_y == y_index:
                                break
                    if point > 0:
                        previous_x_index = x_index
                        previous_y_index = y_index
                lane_cluster += 1

            for i in range(self.grid_y*self.grid_x): #make gt
                temp = temp[temp>-1]
                gt_one = temp.copy()
                if temp[i]>0:
                    gt_one[temp==temp[i]] = 1   #same instance
                    if temp[i] == 0:
                        gt_one[temp!=temp[i]] = 3 #different instance, different class
                    else:
                        gt_one[temp!=temp[i]] = 2 #different instance, same class
                        gt_one[temp==0] = 3 #different instance, different class
                    ground[batch_index][0][i] += gt_one

        return ground

    @staticmethod
    def parse_own_label(labels):
        lane_list = {"Lines": [], "Labels": []}
        for one_line in labels["shapes"]:
            labels = one_line["label"]
            pts = one_line["points"]
            one_line_list = [{"x": pt[0], "y": pt[1]} for pt in pts]
            lane_list["Lines"].append(one_line_list)
            lane_list["Labels"].append(labels)
        assert len(lane_list["Lines"])==len(lane_list["Labels"])
        return lane_list

    @staticmethod
    def load_detect_annot(labels_txt):
        annotations_list = open(labels_txt).readlines()
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_list) == 0:
            return annotations

        for idx, one_label in enumerate(annotations_list):
            one_label = one_label.strip("\n").split(",")
            x1 = int(one_label[0])
            y1 = int(one_label[1])
            x2 = int(one_label[2])
            y2 = int(one_label[3])
            category_id = int(one_label[4])  # 这里0为背景，因此所有非背景目标都从1开始
            width = x2 - x1
            height = y2 - y1

            # some annotations have basically no width / height, skip them
            if width < 1 or height < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = [x1, y1, width, height]
            annotation[0, 4] = category_id - 1  # 这里减一为去掉背景
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    @staticmethod
    def transfer_det_center(annotations,org_shape, target_shape):
        scale_factor = np.array([float(target_shape[0])/org_shape[0],float(target_shape[1])/org_shape[1],
                                float(target_shape[0])/org_shape[0],float(target_shape[1])/org_shape[1]])

        annotations_bboxes = scale_factor.reshape(1,4) * annotations[:,0:4]
        annotations_bboxes[:,0] = np.clip(annotations_bboxes[:,0],a_min=0,a_max=target_shape[0]-1)
        annotations_bboxes[:,2] = np.clip(annotations_bboxes[:,2],a_min=0,a_max=target_shape[0]-1)
        annotations_bboxes[:,1] = np.clip(annotations_bboxes[:,1],a_min=0,a_max=target_shape[1]-1)
        annotations_bboxes[:,3] = np.clip(annotations_bboxes[:,3],a_min=0,a_max=target_shape[1]-1)
        return annotations_bboxes, annotations[:,4]


    @staticmethod
    def cal_split(image,lane_object):
        height , width = image.shape[0], image.shape[1]
        k0_list = []
        k1_list = []
        all_lines = []
        for one_lane in lane_object["Lines"]:
            x_list = []
            y_list = []
            one_line_pts = []
            for pt_index in range(len(one_lane)):
                one_pt = (int(float(one_lane[pt_index]["x"])), height - int(float(one_lane[pt_index]["y"])))
                x_list.append(one_pt[0])
                y_list.append(one_pt[1])
                one_line_pts.append(one_pt)
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    coeff = np.polyfit(x_list, y_list, 1)
                except np.RankWarning:
                    return False,None
                except:
                    return False, None
            k0 = coeff[1]
            k1 = coeff[0]
            k0_list.append(k0)
            k1_list.append(k1)
            all_lines.append(one_line_pts)

        # 进行逻辑判断
        k1_list = np.array(k1_list)
        sorted_k1 = np.sort(k1_list)
        index = np.argsort(k1_list)
        if np.all(sorted_k1>=0) or np.all(sorted_k1) <=0:
            do_split_possible = False
            split_ratio = None
        else:
            index_left_lane = np.where(sorted_k1 <=0)[0][0] # 负得最大的那个为左
            left_lane_index = index[ index_left_lane ]
            right_lane_index = index[-1] # 正得最大的那个为左

            left_lane_pts = np.array(all_lines[left_lane_index])
            right_lane_pts = np.array(all_lines[right_lane_index])

            left_lane_pts_sort = left_lane_pts[ np.argsort((left_lane_pts)[:,1],axis=0)  ]
            right_lane_pts_sort = right_lane_pts[ np.argsort((right_lane_pts)[:,1],axis=0)  ]

            left_x_ = left_lane_pts_sort[0,0]
            right_x_ = right_lane_pts_sort[0,0]
            do_split_possible = True
            split_ratio = (left_x_ + right_x_) / 2.0 / width

        return do_split_possible, split_ratio

    @staticmethod
    def draw_line_on_image(image, lane_object, save_name):
        im_vis_org = image.copy()
        for one_lane in lane_object["Lines"]:
            rd_color = (int(np.random.randint(0, 255)),
                        int(np.random.randint(0, 255)),
                        int(np.random.randint(0, 255)))
            for pt_index in range(len(one_lane) - 1):
                one_pt = one_lane[pt_index]
                one_pt_next = one_lane[pt_index + 1]
                one_pt = (int(float(one_pt["x"])), int(float(one_pt["y"])))
                one_pt_ = (int(float(one_pt_next["x"])), int(float(one_pt_next["y"])))
                print(one_pt)
                cv2.line(im_vis_org, one_pt, one_pt_, rd_color, 3)
        cv2.imwrite(save_name, im_vis_org)

    def draw_label_on_image(self, image, lane_label, obj_label ,seg_label, save_name):
        im_vis_org = image.copy()

        # 语义分割
        np.random.seed(1991)
        seg_arr_vis = np.zeros_like(image)
        for idx in range(len(self.seg_class_list)):
            value = (int(np.random.randint(128,255)),int(np.random.randint(128,255)),int(np.random.randint(128,255)))
            seg_arr_vis[seg_label==idx]=value
        im_vis_org = cv2.addWeighted(im_vis_org,0.5,seg_arr_vis,0.5,0.0)

        # 车道线
        for one_lane in lane_label["Lines"]:
            rd_color = (0,255,0)
            for pt_index in range(len(one_lane) - 1):
                one_pt = one_lane[pt_index]
                one_pt_next = one_lane[pt_index + 1]
                one_pt = (int(float(one_pt["x"])), int(float(one_pt["y"])))
                one_pt_ = (int(float(one_pt_next["x"])), int(float(one_pt_next["y"])))
                print(one_pt)
                cv2.line(im_vis_org, one_pt, one_pt_, rd_color, 3)

        # 目标检测
        for idx,one_box in enumerate(obj_label):
            x1, y1, x2, y2 = int(one_box[0]),int(one_box[1]),int(one_box[2]),int(one_box[3])
            class_category = int(one_box[4]) # 这里非背景类从1开始
            pt1=(x1,y1)
            pt2=(x2,y1)
            pt3=(x2,y2)
            pt4=(x1,y2)
            cv2.line(im_vis_org,pt1,pt2,(0,255,0),2)
            cv2.line(im_vis_org,pt2,pt3,(0,0,255),2)
            cv2.line(im_vis_org,pt3,pt4,(0,0,255),2)
            cv2.line(im_vis_org,pt4,pt1,(0,0,255),2)

            fontScale = 0.5
            thickness = 1
            font = cv2.FONT_HERSHEY_COMPLEX
            pt_txt = (pt1[0], pt1[1] - 5)

            cv2.putText(im_vis_org, str(self.det_class_list[class_category + 1]), pt_txt, font, fontScale,
                        [0, 0, 0], thickness=thickness,lineType=cv2.LINE_AA)

        cv2.imwrite(save_name, im_vis_org)


class Collater(object):
    def __init__(self,
                 target_width,
                 target_height,
                 is_lane=True,
                 is_det=True,
                 is_seg=True):
        self.target_width = target_width
        self.target_height = target_height
        self.is_lane = is_lane
        self.is_det = is_det
        self.is_seg = is_seg

    def __call__(self, batch):
        image_data = np.stack([item["image"] for item in batch]) # images
        image_data = torch.from_numpy(image_data)
        img_shape_list = np.stack([item["src_image_shape"] for item in batch])  # cls
        meta_data = ([item["meta_data"] for item in batch])  # cls


        # =========================================
        # 处理车道线
        # =========================================
        if self.is_lane:
            gt_lane_points_map = np.vstack([item["gt_lane_points_map"] for item in batch]) # location
            gt_lane_points_map = torch.from_numpy(gt_lane_points_map)
            gt_lane_points_instance = np.vstack([item["gt_lane_points_instance"] for item in batch]) # cls
            gt_lane_points_instance = torch.from_numpy(gt_lane_points_instance)
            gt_lane_org_label = ([item["gt_lane_org_label"] for item in batch]) # cls

        else:
            gt_lane_points_map = None
            gt_lane_points_instance = None
            gt_lane_org_label = None

        # =========================================
        # 处理分割
        # =========================================
        if self.is_seg:
            gt_seg = np.stack([cv2.resize(item["gt_seg"],(self.target_width,self.target_height),cv2.INTER_NEAREST)
                               for item in batch]) # seg
            gt_seg = torch.from_numpy(gt_seg)
        else:
            gt_seg=None

        # =========================================
        # 处理检测
        # =========================================
        if self.is_det:
            gt_det_boxes = [item["gt_det_box"] for item in batch] # det boxes
            gt_det_labels = [item["gt_det_label"] for item in batch] # det boxes
        else:
            gt_det_boxes = None
            gt_det_labels = None

        # =========================================
        # 输入
        # =========================================
        output_dict = dict()
        output_dict["image"] = image_data
        output_dict["net_input_image_shape"] = np.stack([item["net_input_image_shape"] for item in batch])
        output_dict["src_image_shape"] = img_shape_list
        output_dict["img_metas"] = meta_data

        if self.is_lane:
            output_dict["annot_lane_path"] = np.stack([item["annot_lane_path"] for item in batch])
            output_dict["annot_lane"] = np.stack([item["annot_lane"] for item in batch])
            output_dict["gt_lane_points_map"] = gt_lane_points_map
            output_dict["gt_lane_points_instance"] = gt_lane_points_instance
            output_dict["gt_lane_org_label"] = gt_lane_org_label

        if self.is_seg:
            output_dict["gt_seg"]=gt_seg

        if self.is_det:
            output_dict["gt_det_bboxes"]=gt_det_boxes
            output_dict["gt_det_labels"]=gt_det_labels

        return output_dict

DEBUG = False
if __name__ == '__main__':
    # 输入测试参数
    CFG_PATH = "../cfgs/ultranet_resnet34.yml"
    MODE = "val"

    cfgs = yaml.safe_load(open(CFG_PATH))
    mt_data = MultitaskData(cfgs=cfgs,mode=MODE)

    trainloader = torch.utils.data.dataloader.DataLoader(mt_data,
                                                         batch_size=2,
                                                         num_workers=0,
                                                         shuffle=True,
                                                         drop_last=False,
                                                         pin_memory=True,
                                                         collate_fn=mt_data.collate_fn)

    one_data = iter(trainloader).__next__()

    for key,value in one_data.items():
        if not isinstance(value,list):
            if value is not None:
                print(key, value.shape)
            else:
                print(key,"None")
        else:
            print(key)
            for elem in value:
                print(elem)