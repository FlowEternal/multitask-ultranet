"""
Function: Model Defination
Author: Zhan Dong Xu
Date: 2021/9/26
"""

from abc import ABC

# torch库
import torch
import torch.nn

# 主干 + neck
from mit.net import MixVisionTransformer, FuseNeck
from mit.resnet import resnet18,resnet34

# 分割
from head_seg.seg_header import SegHeader
from head_seg.segmentation_loss import CrossEntropyLoss

# 检测
from head_detect.centernet import CenterNetHead

# 车道线
from head_lane.lane_header import LanePointHeader

#---------------------------------------------------#
#  Ultranet Defination
#---------------------------------------------------#
class Ultranet(torch.nn.Module, ABC):
    def __init__(self,cfgs,onnx_export = False):
        super(Ultranet, self).__init__()

        # save for reference
        self.cfgs = cfgs
        self.onnx_export = onnx_export

        # general paramters
        self.net_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.net_input_height = self.cfgs["dataloader"]["network_input_height"]
        self.train_seg = self.cfgs["train"]["train_seg"]
        self.train_detect = self.cfgs["train"]["train_detect"]
        self.train_lane = self.cfgs["train"]["train_lane"]
        assert (self.train_seg or self.train_detect or self.train_lane) == True

        # backbone paramters
        self.choose_resnet = cfgs["backbone"]["choose_resnet"]
        self.resnet_type = cfgs["backbone"]["resnet_type"]

        self.in_channels = cfgs["backbone"]["in_channels"]
        self.embed_dims = cfgs["backbone"]["embed_dims"]
        self.num_stages = cfgs["backbone"]["num_stages"]
        self.num_layers = cfgs["backbone"]["num_layers"]
        self.num_heads = cfgs["backbone"]["num_heads"]
        self.patch_sizes = cfgs["backbone"]["patch_sizes"]
        self.strides = cfgs["backbone"]["strides"]
        self.sr_ratios = cfgs["backbone"]["sr_ratios"]
        self.out_indices = tuple(cfgs["backbone"]["out_indices"])
        self.mlp_ratio = cfgs["backbone"]["mlp_ratio"]
        self.qkv_bias = cfgs["backbone"]["qkv_bias"]
        self.drop_rate = cfgs["backbone"]["drop_rate"]
        self.attn_drop_rate = cfgs["backbone"]["attn_drop_rate"]
        self.drop_path_rate = cfgs["backbone"]["drop_path_rate"]
        self.act_cfg = cfgs["backbone"]["act_cfg"]
        self.norm_cfg = cfgs["backbone"]["norm_cfg"]

        # neck parameters
        self.interpolate_mode = cfgs["backbone"]["interpolate_mode"]
        self.in_channels_ = cfgs["backbone"]["in_channels_"]
        self.in_index = cfgs["backbone"]["in_index"]
        self.channels = cfgs["backbone"]["channels"]
        self.act_cfg_ = cfgs["backbone"]["act_cfg_"]
        self.norm_cfg_ = cfgs["backbone"]["norm_cfg_"]
        self.align_corners = cfgs["backbone"]["align_corners"]
        self.neck_scale = cfgs["backbone"]["neck_scale"]

        if not self.choose_resnet:
            # construct vison transformer backbone
            self.backbone = MixVisionTransformer(
                in_channels=self.in_channels,
                embed_dims=self.embed_dims,
                num_stages=self.num_stages,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                patch_sizes=self.patch_sizes,
                strides=self.strides,
                sr_ratios=self.sr_ratios,
                out_indices=self.out_indices,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=self.drop_path_rate,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
            )

        else:
            if self.resnet_type == 18:
                self.backbone = resnet18(pretrained=False)

            else:
                self.backbone = resnet34(pretrained=False)

        # construct fusion neck
        self.neck = FuseNeck(
            interpolate_mode=self.interpolate_mode,
            in_channels=self.in_channels_,
            in_index=self.in_index,
            channels=self.channels,
            act_cfg=self.act_cfg_,
            norm_cfg=self.norm_cfg_,
            align_corners=self.align_corners,
            neck_scale=self.neck_scale
        )

        # 1.load segmentation header
        if self.train_seg:
            self.segment_class_list = self.cfgs["segment"]["class_list"]
            self.weight_seg = self.cfgs["segment"]["class_weight"]
            assert len(self.segment_class_list) == len(self.weight_seg)

            self.use_top_k = self.cfgs["segment"]["use_top_k"]
            self.top_k_ratio = self.cfgs["segment"]["top_k_ratio"]
            self.use_focal = self.cfgs["segment"]["use_focal"]

            # 分割头
            self.seg_header = SegHeader(self.cfgs)

            # 分割损失函数
            self.loss_seg = CrossEntropyLoss(
                class_weights=torch.tensor(self.weight_seg),
                use_top_k=self.use_top_k,
                top_k_ratio=self.top_k_ratio,
                use_focal=self.use_focal
            )


        # 2.load detection header
        if self.train_detect:
            self.class_list = cfgs["detection"]["class_list"][1:]
            self.num_classes = cfgs["detection"]["num_classes"]
            assert len(self.class_list) == self.num_classes

            self.in_channel = cfgs["detection"]["in_channel"]
            self.feat_channel = cfgs["detection"]["feat_channel"]

            # 各项权重
            self.loss_center_heatmap = cfgs["detection"]["loss_center_heatmap"]
            self.loss_wh = cfgs["detection"]["loss_wh"]
            self.loss_offset = cfgs["detection"]["loss_offset"]

            # 后处理参数
            self.test_cfg = cfgs["detection"]["test_cfg"]

            # 检测头
            self.center_head = CenterNetHead(
                in_channel=self.in_channel,
                feat_channel=self.feat_channel,
                num_classes=self.num_classes,
                loss_center_heatmap=self.loss_center_heatmap,
                loss_wh=self.loss_wh,
                loss_offset=self.loss_offset,
                train_cfg=None,
                test_cfg=self.test_cfg,
                init_cfg=None
            )

            # 检测损失函数
            self.loss_det = self.center_head.loss


        # 3.load lane header
        if self.train_lane:
            self.cluster_feat_dim = cfgs["lane"]["cluster_feat_dim"]
            self.exist_condidence_loss = cfgs["lane"]["exist_condidence_loss"]
            self.nonexist_confidence_loss = cfgs["lane"]["nonexist_confidence_loss"]
            self.offset_loss = cfgs["lane"]["offset_loss"]
            self.sisc_loss = cfgs["lane"]["sisc_loss"]
            self.disc_loss = cfgs["lane"]["disc_loss"]

            self.k1 = cfgs["lane"]["k1"]
            self.thresh = cfgs["lane"]["thresh"]
            self.threshold_instance = cfgs["lane"]["threshold_instance"]
            self.resize_ratio = cfgs["lane"]["resize_ratio"]
            self.grid_x = int(self.net_input_width / self.resize_ratio)
            self.grid_y = int(self.net_input_height / self.resize_ratio)
            self.x_size = self.net_input_width
            self.y_size = self.net_input_height

            # 车道线头
            self.lane_header = LanePointHeader(
                  self.channels,
                  self.cluster_feat_dim,
                  grid_x=self.grid_x,
                  grid_y=self.grid_y,
                  x_size=self.x_size,
                  y_size=self.y_size,
                  k1=self.k1,
                  thresh=self.thresh,
                  threshold_instance=self.threshold_instance,
                  resize_ratio=self.resize_ratio
            )

            self.loss_lane = self.lane_header.cal_loss


    def forward(self, x, mode = "train"):
        # feature abstraction
        if self.choose_resnet:
            feats = self.backbone(x, True)[1:]
        else:
            feats = self.backbone(x)
        fused_feats = self.neck(feats)

        output_dict = {}
        # segment
        if self.train_seg:
            output_seg = self.seg_header(fused_feats)
            output_dict.update({"seg":output_seg})
        else:
            output_seg = None

        # detection
        if self.train_detect:
            outs_detect = self.center_head([fused_feats])
            output_dict.update({"detection":outs_detect})
        else:
            outs_detect = None

        # lane
        if self.train_lane:
            outs_lane = self.lane_header(fused_feats)
            output_dict.update({"lane":outs_lane})
        else:
            outs_lane = None

        if mode !="deploy":
            return output_dict
        else:
            # post process seg
            output_seg = torch.argmax(output_seg, dim=1)

            # post process detect
            center_heatmap_pred, wh_pred, offset_pred = outs_detect
            hmax = torch.nn.functional.max_pool2d(center_heatmap_pred[0], 3, stride=1, padding=1)
            heat_point = (hmax == center_heatmap_pred[0]).float() * center_heatmap_pred[0]
            topk_scores, topk_inds = torch.topk(heat_point.view(1, -1), 100) # 这里取top 100

            out_confidence, out_offset, out_instance = outs_lane
            return output_seg,\
                   topk_scores, topk_inds, wh_pred[0], offset_pred[0],\
                   out_confidence, out_offset, out_instance

    # 关键函数 -- loss计算
    def cal_loss(self, pred_dict, gt_dict):
        loss_dict = {}

        if self.train_seg:
            gt_seg = gt_dict["gt_seg"]
            output_seg = pred_dict["seg"]
            loss_seg = self.loss_seg(output_seg, gt_seg.long())
            if loss_seg == 0 or not torch.isfinite(loss_seg):
                print("cal segment loss diverge!")
                exit()

            loss_dict.update({"loss_seg":loss_seg})


        if self.train_detect:
            target_shape = (self.net_input_width, self.net_input_height)
            gt_bboxes = gt_dict["gt_det_bboxes"]
            gt_labels = gt_dict["gt_det_labels"]
            loss_cal_inputy = pred_dict["detection"] + (gt_bboxes, gt_labels, target_shape)
            loss_dict_detection = self.loss_det(*loss_cal_inputy)
            loss_dict["loss_detect_center_heatmap"] = loss_dict_detection["loss_center_heatmap"]
            loss_dict["loss_detect_wh"] = loss_dict_detection["loss_wh"]
            loss_dict["loss_detect_offset"] = loss_dict_detection["loss_offset"]


        if self.train_lane:
            outs_lane = pred_dict["lane"]
            gt_lane_points_map = gt_dict["gt_lane_points_map"]
            gt_lane_points_instance = gt_dict["gt_lane_points_instance"]
            exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, disc_loss = \
                self.loss_lane(gt_lane_points_map, gt_lane_points_instance, outs_lane)

            loss_dict["exist_condidence_loss"] = exist_condidence_loss
            loss_dict["nonexist_confidence_loss"] = nonexist_confidence_loss
            loss_dict["offset_loss"] = offset_loss
            loss_dict["sisc_loss"] = sisc_loss
            loss_dict["disc_loss"] = disc_loss

        return loss_dict

if __name__ == '__main__':
    import yaml
    CFG_PATH = "cfgs/ultranet_resnet34.yml"
    cfgs = yaml.safe_load(open(CFG_PATH))
    ultranet = Ultranet(cfgs=cfgs).cuda()

    # inference
    batch_size = 6
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    dummy_input = torch.randn((batch_size, 3, net_input_height, net_input_width)).cuda()

    # test speed
    import time
    for _ in range(50):
        ouptut = ultranet(dummy_input)

    avg_runtime = 0.0
    for _ in range(100):
        tic = time.time()
        ouptut = ultranet(dummy_input)
        torch.cuda.synchronize()
        print("inference time: %i" %(1000*(time.time() - tic)))
        avg_runtime += 1000*(time.time() - tic)
    print("average time: %i" % (avg_runtime/100))

