from abc import ABC

import numpy as np
import json
import cv2
import csaps
import torch
import torch.nn as nn

from head_lane.evaluation import LaneEval


def write_result_json(result_data, x, y, testset_index):
    for index, batch_idx in enumerate(testset_index):
        for i in x[index]:
            result_data[batch_idx]['lanes'].append(i)
            result_data[batch_idx]['run_time'] = 1
    return result_data


def fitting(x, y, target_h, org_width):
    out_x = []
    out_y = []
    count = 0
    x_size = org_width

    if len(x[0]) ==0:
        return [[]],[]

    for x_batch, y_batch in zip(x, y):
        predict_x_batch = []
        predict_y_batch = []
        for i, j in zip(x_batch, y_batch):
            min_y = min(j)
            max_y = max(j)
            temp_x = []
            temp_y = []

            jj = []
            pre = -100
            for temp in j[::-1]:
                if temp > pre:
                    jj.append(temp)
                    pre = temp
                else:
                    jj.append(pre + 0.00001)
                    pre = pre + 0.00001
            sp = csaps.CubicSmoothingSpline(jj, i[::-1], smooth=0.0001)

            last = 0
            last_second = 0
            last_y = 0
            last_second_y = 0
            for h in target_h[count]:
                temp_y.append(h)
                if h < min_y:
                    temp_x.append(-2)
                elif min_y <= h and h <= max_y:
                    temp_x.append(sp([h])[0])
                    last = temp_x[-1]
                    last_y = temp_y[-1]
                    if len(temp_x) < 2:
                        last_second = temp_x[-1]
                        last_second_y = temp_y[-1]
                    else:
                        last_second = temp_x[-2]
                        last_second_y = temp_y[-2]
                else:
                    if last < last_second:
                        l = int(last_second - float(-last_second_y + h) * abs(last_second - last) / abs(
                            last_second_y + 0.0001 - last_y))
                        if l > x_size or l < 0:
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
                    else:
                        l = int(last_second + float(-last_second_y + h) * abs(last_second - last) / abs(
                            last_second_y + 0.0001 - last_y))
                        if l > x_size or l < 0:
                            temp_x.append(-2)
                        else:
                            temp_x.append(l)
            predict_x_batch.append(temp_x)
            predict_y_batch.append(temp_y)
        out_x.append(predict_x_batch)
        out_y.append(predict_y_batch)
        count += 1

    return out_x, out_y


def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

def convert_to_original_size(x, y, ratio_w, ratio_h):
    # convert results to original size
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        out_x.append((np.array(i) / ratio_w).tolist())
        out_y.append((np.array(j) / ratio_h).tolist())

    return out_x, out_y

class Conv2D_BatchNorm_Relu(nn.Module, ABC):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True, dilation=1):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size,
                                                    padding=padding, stride=stride, bias=bias, dilation=dilation),
                                    nn.BatchNorm2d(n_filters),
                                    #nn.ReLU(inplace=True),)
                                    nn.PReLU(),)
        else:
            self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Output(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size//2, 3, 1, 1, dilation=1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size//2, in_size//4, 3, 1, 1, dilation=1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size//4, out_size, 1, 0, 1, acti = False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class LanePointHeader(nn.Module):
    def __init__(self,
                 feat_channel,
                 cluster_feat_dim,
                 grid_x=80,
                 grid_y=80,
                 x_size=640,
                 y_size=640,
                 k1=1.0,
                 thresh=0.8,
                 threshold_instance=0.08,
                 resize_ratio=8):
        super(LanePointHeader, self).__init__()
        self.out_confidence = Output(feat_channel, 1)
        self.out_offset = Output(feat_channel, 2)
        self.out_instance = Output(feat_channel, cluster_feat_dim)

        self.grid_x = grid_x
        self.grid_y = grid_y

        self.x_size = x_size
        self.y_size = y_size

        self.K1 = k1
        self.thresh = thresh
        self.threshold_instance = threshold_instance
        self.feature_size = cluster_feat_dim
        self.resize_ratio = resize_ratio

        grid_location = np.zeros((self.grid_y, self.grid_x, 2))
        for y in range(self.grid_y):
            for x in range(self.grid_x):
                grid_location[y][x][0] = x
                grid_location[y][x][1] = y
        self.grid_location = grid_location

        # 直接定死
        self.color = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                 (255, 255, 255), (100, 255, 0), (100, 0, 255), (255, 100, 0), (0, 100, 255), (255, 0, 100),
                 (0, 255, 100)]

    def forward(self, feats):
        out_confidence = self.out_confidence(feats)
        out_offset = self.out_offset(feats)
        out_instance = self.out_instance(feats)
        return [out_confidence, out_offset, out_instance]


    def cal_loss(self,ground_truth_point, ground_truth_instance, result):
        # update lane_detection_network
        exist_condidence_loss = 0
        nonexist_confidence_loss = 0
        offset_loss = 0
        sisc_loss = 0
        disc_loss = 0

        # hard sampling ##################################################################
        real_batch_size = result[0].shape[0]


        for (confidance, offset, feature) in [result]:
            # compute loss for point prediction

            # exist confidance loss##########################
            # confidance = torch.sigmoid(confidance)
            confidance_gt = ground_truth_point[:, 0, :, :]
            confidance_gt = confidance_gt.view(real_batch_size, 1, self.grid_y, self.grid_x)
            exist_condidence_loss = exist_condidence_loss + \
                                    torch.sum((1 - confidance[confidance_gt == 1]) ** 2) / \
                                    torch.sum(confidance_gt == 1)

            # non exist confidance loss##########################
            target = confidance[confidance_gt == 0]
            nonexist_confidence_loss = nonexist_confidence_loss + \
                                       torch.sum((target[target > 0.01]) ** 2) / \
                                       (torch.sum(target > 0.01) + 1)

            # offset loss ##################################
            offset_x_gt = ground_truth_point[:, 1:2, :, :]
            offset_y_gt = ground_truth_point[:, 2:3, :, :]

            predict_x = offset[:, 0:1, :, :]
            predict_y = offset[:, 1:2, :, :]

            offset_loss = offset_loss + \
                          torch.sum((offset_x_gt[confidance_gt == 1] - predict_x[confidance_gt == 1]) ** 2) / \
                          torch.sum(confidance_gt == 1) + \
                          torch.sum((offset_y_gt[confidance_gt == 1] - predict_y[confidance_gt == 1]) ** 2) / \
                          torch.sum(confidance_gt == 1)

            # compute loss for similarity #################
            feature_map = feature.view(real_batch_size, self.feature_size, 1, self.grid_y * self.grid_x)
            feature_map = feature_map.expand(real_batch_size, self.feature_size, self.grid_y * self.grid_x,
                                             self.grid_y * self.grid_x)  # .detach()

            point_feature = feature.view(real_batch_size, self.feature_size, self.grid_y * self.grid_x, 1)
            point_feature = point_feature.expand(real_batch_size, self.feature_size, self.grid_y * self.grid_x,
                                                 self.grid_y * self.grid_x)  # .detach()

            distance_map = (feature_map - point_feature) ** 2
            distance_map = torch.sum(distance_map, dim=1).view(real_batch_size, 1, self.grid_y * self.grid_x,
                                                               self.grid_y * self.grid_x)

            # same instance
            sisc_loss = sisc_loss + \
                        torch.sum(distance_map[ground_truth_instance == 1]) / \
                        torch.sum(ground_truth_instance == 1)

            # different instance, same class
            disc_loss = disc_loss + \
                        torch.sum((self.K1 - distance_map[ground_truth_instance == 2])[
                                      (self.K1 - distance_map[ground_truth_instance == 2]) > 0]) / \
                        torch.sum(ground_truth_instance == 2)

        return exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, disc_loss


    def draw_points(self, x, y, image,ratio_w,ratio_h):
        color_index = 0
        for i, j in zip(x, y):
            color_index += 1
            if color_index > 12:
                color_index = 12
            if len(i) > 6:
                for index in range(len(i)):
                    pt_x = int(i[index]/ratio_w)
                    pt_y = int(j[index]/ratio_h)
                    image = cv2.circle(image, (pt_x, pt_y), 7, self.color[color_index], -1)

        return image

    def display(self,out_x, out_y,images, ratio_w, ratio_h):
        num_batch = len(images)
        out_images = images
        for i in range(num_batch):
            in_x = out_x[i]
            in_y = out_y[i]

            out_images[i] = self.draw_points(in_x, in_y, images[i],ratio_w,ratio_h)

        return out_images

    def decode_result(self,result):
        confidences, offsets, instances = result
        num_batch = confidences.shape[0]
        out_x = []
        out_y = []
        for batch_index in range(num_batch):
            confidence = confidences[batch_index].view(self.grid_y, self.grid_x).cpu().data.numpy()
            offset = offsets[batch_index].cpu().data.numpy()
            offset = np.rollaxis(offset, axis=2, start=0)
            offset = np.rollaxis(offset, axis=2, start=0)

            instance = instances[batch_index].cpu().data.numpy()
            instance = np.rollaxis(instance, axis=2, start=0)
            instance = np.rollaxis(instance, axis=2, start=0)

            # generate point and cluster
            raw_x, raw_y = self.generate_result(confidence, offset, instance, self.thresh)

            # eliminate fewer points
            in_x, in_y = self.eliminate_fewer_points(raw_x, raw_y)

            # sort points along y
            in_x, in_y = self.sort_along_y(in_x, in_y)

            out_x.append(in_x)
            out_y.append(in_y)
        return out_x, out_y


    def generate_result(self,confidance, offsets, instance, thresh):

        mask = confidance > thresh

        grid = self.grid_location[mask]

        offset = offsets[mask]
        feature = instance[mask]

        lane_feature = []
        x = []
        y = []
        for i in range(len(grid)):
            if (np.sum(feature[i] ** 2)) >= 0:
                point_x = int((offset[i][0] + grid[i][0]) * self.resize_ratio)
                point_y = int((offset[i][1] + grid[i][1]) * self.resize_ratio)
                if point_x > self.x_size or point_x < 0 or point_y > self.y_size or point_y < 0:
                    continue
                if len(lane_feature) == 0:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                else:
                    min_feature_index = -1
                    min_feature_dis = 10000
                    for feature_idx, j in enumerate(lane_feature):
                        dis = np.linalg.norm((feature[i] - j) ** 2)
                        if min_feature_dis > dis:
                            min_feature_dis = dis
                            min_feature_index = feature_idx
                    if min_feature_dis <= self.threshold_instance:
                        lane_feature[min_feature_index] = (lane_feature[min_feature_index] * len(x[min_feature_index]) +
                                                           feature[i]) / (len(x[min_feature_index]) + 1)
                        x[min_feature_index].append(point_x)
                        y[min_feature_index].append(point_y)
                    elif len(lane_feature) < 12:
                        lane_feature.append(feature[i])
                        x.append([point_x])
                        y.append([point_y])

        return x, y

    @staticmethod
    def eliminate_fewer_points(x, y):
        # eliminate fewer points
        out_x = []
        out_y = []
        for i, j in zip(x, y):
            if len(i) > 2:
                out_x.append(i)
                out_y.append(j)
        return out_x, out_y

    @staticmethod
    def sort_along_y(x, y):
        out_x = []
        out_y = []

        for i, j in zip(x, y):
            i = np.array(i)
            j = np.array(j)

            ind = np.argsort(j, axis=0)
            out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
            out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())

        return out_x, out_y




if __name__ == '__main__':
    import yaml
    CFG_PATH = "../cfgs/ultranet.yml"
    cfgs = yaml.safe_load(open(CFG_PATH))

    # input
    channels = cfgs["backbone"]["channels"]
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    dummy_input = torch.randn((2, channels, int(net_input_height/8), int(net_input_width/8) )).cuda()

    # paramter for detect head
    cluster_feat_dim = cfgs["lane"]["cluster_feat_dim"]
    exist_condidence_loss = cfgs["lane"]["exist_condidence_loss"]
    nonexist_confidence_loss = cfgs["lane"]["nonexist_confidence_loss"]
    offset_loss = cfgs["lane"]["offset_loss"]
    sisc_loss = cfgs["lane"]["sisc_loss"]
    disc_loss = cfgs["lane"]["disc_loss"]

    k1 = cfgs["lane"]["k1"]
    thresh = cfgs["lane"]["thresh"]
    threshold_instance = cfgs["lane"]["threshold_instance"]
    resize_ratio = cfgs["lane"]["resize_ratio"]
    grid_x = int(net_input_width/resize_ratio)
    grid_y = int(net_input_height/resize_ratio)
    x_size = net_input_width
    y_size = net_input_height


    lane_header = LanePointHeader(channels,
                                  cluster_feat_dim,
                                  grid_x=grid_x,
                                  grid_y=grid_y,
                                  x_size=x_size,
                                  y_size=y_size,
                                  k1=k1,
                                  thresh=thresh,
                                  threshold_instance=threshold_instance,
                                  resize_ratio=resize_ratio).cuda()

    # =========================================
    # inference test
    # =========================================
    outs = lane_header(dummy_input)
    print(outs[0].shape)
    print(outs[1].shape)
    print(outs[2].shape)

    # =========================================
    # training
    # =========================================
    dummy_label_conf_offset = torch.randn((2, 3, int(net_input_height/8), int(net_input_width/8) )).cuda()
    conf_ = torch.zeros(int(net_input_height/8), int(net_input_width/8)).cuda()
    conf_[3,3] = 1.0
    conf_[13,13] = 1.0
    conf_[33,33] = 1.0
    conf_[70,70] = 1.0
    dummy_label_conf_offset[0,0] = conf_
    dummy_label_conf_offset[1,0] = conf_
    dummy_label_instance = torch.ones((2, 1, int(net_input_height/8)*int(net_input_width/8), int(net_input_height/8)*int(net_input_width/8) ) ).cuda()
    loss_all = lane_header.cal_loss(dummy_label_conf_offset,dummy_label_instance,outs)

    for loss_item in loss_all:
        print(loss_item)

    # =========================================
    # demo
    # =========================================
    images = [np.ones([640,640,3], dtype=np.uint8),np.ones([640,640,3], dtype=np.uint8)]
    out_x_, out_y_ = lane_header.decode_result(outs)
    images = lane_header.display(out_x_,out_y_,images)

    # =========================================
    # validation
    # =========================================
    # generate test_label.json
    tmp_dict = {"lanes": [
        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 648, 636, 626, 615, 605, 595, 585, 575, 565, 554, 545, 536, 526, 517,
         508, 498, 489, 480, 470, 461, 452, 442, 433, 424, 414, 405, 396, 386, 377, 368, 359, 349, 340, 331, 321, 312,
         303, 293, 284, 275, 265, 256, 247, 237, 228, 219],
        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 681, 692, 704, 716, 728, 741, 754, 768, 781, 794, 807, 820, 834, 847,
         860, 873, 886, 900, 913, 926, 939, 952, 966, 979, 992, 1005, 1018, 1032, 1045, 1058, 1071, 1084, 1098, 1111,
         1124, 1137, 1150, 1164, 1177, 1190, 1203, 1216, 1230, 1243, 1256, 1269],
        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 713, 746, 778, 811, 845, 880, 916, 951, 986, 1022, 1057, 1092, 1128,
         1163, 1198, 1234, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
         -2, -2, -2, -2, -2, -2, -2],
        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 754, 806, 858, 909, 961, 1013, 1064, 1114, 1164, 1213, 1263, -2, -2,
         -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
         -2, -2, -2, -2, -2]],
     "h_samples": [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350,
                   360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550,
                   560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710],
     "raw_file": "img/test_0.png"}

    json_list = []
    for i in range(20):
        tmp__ = tmp_dict.copy()
        tmp__["raw_file"] = "img/test_%i.png" %i
        json_list.append(tmp__)
    lane_header.save_result(json_list, "test_label.json")

    # eval 数据 人为构造
    tmp_data = dict()
    tmp_data["h_samples"] = list(range(160,720,10))
    tmp_data["lanes"] = []
    tmp_data["run_time"] = 1
    tmp_data["raw_file"] = "img/test.png"
    result_data = list()
    for i in range(20):
        tmp_data_ = tmp_data.copy()
        tmp_data_["raw_file"] = "img/test_%i.png" %i
        result_data.append(tmp_data_)

    for k in range(10):
        outs = lane_header(dummy_input)
        x, y = lane_header.decode_result(outs)
        ratio_w = net_input_width / 1280.0
        ratio_h = net_input_height / 720.0
        target_h = [np.array(range(160,720,10)),np.array(range(160,720,10))]
        testset_index = [0 + 2 * k,1 + 2 * k]

        x_ = []
        y_ = []
        for i, j in zip(x, y):
            temp_x, temp_y = convert_to_original_size(i, j, ratio_w, ratio_h)
            x_.append(temp_x)
            y_.append(temp_y)
        x_, y_ = fitting(x_, y_, target_h, 1280)
        result_data = write_result_json(result_data, x_, y_, testset_index)

    save_result(result_data, "test_result.json")

    print(LaneEval.bench_one_submit("test_result.json", "test_label.json"))


