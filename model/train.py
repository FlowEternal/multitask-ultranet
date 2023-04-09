"""
Function: Ultranet Training Scripts
Author: Zhan Dong Xu
Date: 2021/9/24
"""

# 通用库导入
import os, yaml, time, shutil
import numpy as np
import prettytable as pt
import warnings
warnings.filterwarnings("ignore")

# 数据loader导入
from dataset.dataloader import MultitaskData

# 模型导入
from model import Ultranet
import torch.distributed
import torch.utils.data.dataloader

# 分割相关库
from head_seg.seg_metrics import IntersectionOverUnion

# 检测相关库
from head_detect.centernet import bbox2result
from head_detect.gen_val_json import gen_coco_label
from head_detect.coco import CocoDataset

# 车道线相关库
from head_lane.lane_header import convert_to_original_size,fitting,save_result
from head_lane.evaluation import LaneEval


class UltraTrainer(object):
    def __init__(self, cfgs, cfg_path):
        self.cfgs = cfgs
        self.tag = self.cfgs["tag"]
        self.logs = self.cfgs["train"]["logs"]
        self.print_interval = self.cfgs["train"]["print_interval"]

        # 保存路径
        self.save_dir = os.path.join(self.logs, time.strftime('%d_%B_%Y_%H_%M_%S_%Z') + "_" + self.tag)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

        # 配置文件
        self.cfg_path = cfg_path
        self.cfg_path_backup = os.path.join(self.save_dir,"config.yml")
        shutil.copy(self.cfg_path, self.cfg_path_backup)

        # 模型文件保存路径
        self.model_save_dir = os.path.join(self.save_dir,"model")
        if not os.path.exists(self.model_save_dir): os.makedirs(self.model_save_dir)

        # 并行训练初始化相关
        self.use_distribute = self.cfgs["train"]["use_distribute"]
        if self.use_distribute:
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='tcp://localhost:23457',
                                                 rank=0,
                                                 world_size=1)


        # 训练任务
        self.train_detect = self.cfgs["train"]["train_detect"]
        self.train_seg = self.cfgs["train"]["train_seg"]
        self.train_lane = self.cfgs["train"]["train_lane"]
        assert (self.train_seg or self.train_detect or self.train_lane) == True

        # 微调相关变量
        self.flag_joint = True
        self.flag_lane = False
        self.flag_seg = False
        self.flag_det = False

        #---------------------------------------------------#
        #  1.数据加载模块
        #---------------------------------------------------#
        self.batch_size_train = self.cfgs["train"]["batch_size_train"]
        self.num_worker_train = self.cfgs["train"]["num_worker_train"]

        self.batch_size_valid = self.cfgs["train"]["batch_size_valid"]
        self.num_worker_valid = self.cfgs["train"]["num_worker_valid"]

        self.net_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.net_input_height = self.cfgs["dataloader"]["network_input_height"]

        # training loader
        self.train_data = MultitaskData(cfgs=cfgs, mode="train")
        self.trainloader = torch.utils.data.dataloader.DataLoader(
                                                             self.train_data,
                                                             batch_size=self.batch_size_train,
                                                             num_workers=self.num_worker_train,
                                                             shuffle=True,
                                                             drop_last=False,
                                                             pin_memory=True,
                                                             collate_fn=self.train_data.collate_fn,
                                                             )

        # testing loader
        self.valid_data = MultitaskData(cfgs=cfgs, mode="val")
        self.validloader = torch.utils.data.dataloader.DataLoader(
                                                             self.valid_data,
                                                             batch_size=self.batch_size_valid,
                                                             num_workers=self.num_worker_valid,
                                                             shuffle=False,
                                                             drop_last=False,
                                                             pin_memory=True,
                                                             collate_fn=self.valid_data.collate_fn,
                                                             )

        #---------------------------------------------------#
        #  2.模型加载模块
        #---------------------------------------------------#
        self.ultranet = Ultranet(cfgs=cfgs).cuda()
        self.continue_train = self.cfgs["train"]["continue_train"]
        self.weight_file = self.cfgs["train"]["weight_file"]
        if self.continue_train:
            def deparallel_model(dict_param):
                ck_dict_new = dict()
                for key, value in dict_param.items():
                    temp_list = key.split(".")[1:]
                    new_key = ""
                    for tmp in temp_list:
                        new_key += tmp + "."
                    ck_dict_new[new_key[0:-1]] = value
                return ck_dict_new

            dict_old = torch.load(self.weight_file)
            if self.use_distribute:
                dict_new = deparallel_model(dict_old)
            else:
                dict_new = dict_old
            self.ultranet.load_state_dict(dict_new,strict=False)

        # 并行训练开启与否
        if self.use_distribute:
            self.ultranet = torch.nn.parallel.DistributedDataParallel(self.ultranet, find_unused_parameters=True)

        #---------------------------------------------------#
        #  3.优化器 + loss权重
        #---------------------------------------------------#
        self.epoch = self.cfgs["train"]["epoch"]
        self.lr = self.cfgs["train"]["lr"]
        self.weight_decay = self.cfgs["train"]["weight_decay"]
        self.total_iters = len(self.trainloader) * self.epoch
        self.optimizer = torch.optim.Adam(self.ultranet.parameters(), self.lr, weight_decay= self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_iters, eta_min=1e-8)

        self.segment_weight = self.cfgs["segment"]["segment_weight"]
        self.detection_weight = self.cfgs["detection"]["detection_weight"]
        self.lane_weight = self.cfgs["lane"]["lane_weight"]

        #---------------------------------------------------#
        #  4.模型验证 -- 关键
        #---------------------------------------------------#
        if self.train_seg:
            self.segment_class_list = self.cfgs["segment"]["class_list"]
            self.metric_evaluator_iou = IntersectionOverUnion(n_classes=len(self.segment_class_list))

        if self.train_detect:
            self.num_class_detect = cfgs["detection"]["num_classes"]
            self.class_list = cfgs["detection"]["class_list"][1:]
            self.eval_kwargs = {'interval': 1, 'metric': 'bbox'}

            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                self.eval_kwargs.pop(key, None)
            self.eval_kwargs.update(dict(metric=["bbox"], **{}))
            self.root_dir = self.cfgs["dataloader"]["data_list"].replace("/list","")

            # 准备gt_detect_label.json
            eval_dir = os.path.join(self.root_dir, "eval_detect")
            if not os.path.exists(eval_dir): os.makedirs(eval_dir)
            self.val_gt_json = gen_coco_label(self.root_dir)  # 产生真值json

        if self.train_lane:
            self.gt_datas = []
            self.first_time = True

    def cal_total_loss(self, loss_dict):
        loss_total = 0.0
        if self.train_seg:
            loss_total += loss_dict["loss_seg"] * self.segment_weight

        if self.train_detect:
            loss_total += (loss_dict["loss_detect_center_heatmap"]  +
                           loss_dict["loss_detect_wh"]  +
                           loss_dict["loss_detect_offset"] ) * self.detection_weight

        if self.train_lane:
            loss_total +=  (loss_dict["exist_condidence_loss"] +
                            loss_dict["nonexist_confidence_loss"] +
                            loss_dict["offset_loss"] +
                            loss_dict["sisc_loss"] +
                            loss_dict["disc_loss"]) * self.lane_weight

            return loss_total

    def print_loss_info(self, loss_dict, epoch, batch_idx, mode="train"):
        if mode == "train":
            print("TRAIN Epoch [%i|%i] Iter [%i|%i] Lr %.5f" % (epoch , self.epoch,
                                                                batch_idx , len(self.trainloader),
                                                                self.optimizer.param_groups[0]["lr"]))
        else:
            print("VALID Epoch [%i|%i] Iter [%i|%i] Lr %.5f" % (epoch , self.epoch,
                                                                batch_idx , len(self.validloader),
                                                                self.optimizer.param_groups[0]["lr"]))

        tb = pt.PrettyTable()
        row_list = list()
        key_list = list()
        for key, value in loss_dict.items():
            value_str = float("%.3f" %value.item())
            row_list.append(value_str)
            key_list.append(key)

        tb.field_names = key_list
        tb.add_row(row_list)
        print(tb)
        print()

    def to_gpu(self,batch_data):
        batch_data["image"] = batch_data["image"].cuda().float()
        if self.train_lane:
            batch_data["gt_lane_points_map"] = batch_data["gt_lane_points_map"].cuda().float()
            batch_data["gt_lane_points_instance"] = batch_data["gt_lane_points_instance"].cuda().float()

        if self.train_seg:
            batch_data["gt_seg"] = batch_data["gt_seg"].cuda().float()

        if self.train_detect:
            for i in range(len(batch_data["gt_det_bboxes"])):
                batch_data["gt_det_bboxes"][i] = torch.tensor(batch_data["gt_det_bboxes"][i]).cuda().float()
                batch_data["gt_det_labels"][i] = torch.tensor(batch_data["gt_det_labels"][i]).cuda().long()

        return batch_data

    def train_one_epoch(self, epoch):
        self.ultranet.train()
        for iter_idx, batch_data in enumerate(self.trainloader):

            # forward pass
            batch_data = self.to_gpu(batch_data)
            inputs = batch_data["image"]
            outputs = self.ultranet(inputs)
            if self.use_distribute:
                loss_dict = self.ultranet.module.cal_loss(outputs, batch_data)
            else:
                loss_dict = self.ultranet.cal_loss(outputs, batch_data)

            loss_total = self.cal_total_loss(loss_dict)
            loss_dict.update({"total_loss": loss_total})

            # 打印结果
            if iter_idx % self.print_interval ==0:
                self.print_loss_info(loss_dict, epoch, iter_idx,mode="train")

            self.optimizer.zero_grad()
            loss_total.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            self.optimizer.step()

            # 调整学习率
            self.scheduler.step()

        return

    def valid(self, epoch):
        self.ultranet.eval()
        detect_result = list()
        lane_result = list()
        for iter_idx , batch_data in enumerate(self.validloader):
            # forward pass
            batch_data = self.to_gpu(batch_data)
            inputs = batch_data["image"]
            batch_size_tmp = inputs.shape[0]
            outputs = self.ultranet(inputs)
            if self.use_distribute:
                loss_dict = self.ultranet.module.cal_loss(outputs, batch_data)
            else:
                loss_dict = self.ultranet.cal_loss(outputs, batch_data)

            loss_total = self.cal_total_loss(loss_dict)
            loss_dict.update({"total_loss": loss_total})

            # 打印结果
            self.print_loss_info(loss_dict, epoch, iter_idx, mode="valid")

            if self.train_seg:
                #---------------------------------------------------#
                #  分割部分metric
                #---------------------------------------------------#
                output_seg = outputs["seg"]
                gt_seg = batch_data["gt_seg"]

                for batch_idx in range(batch_size_tmp):
                    gt_seg_one = gt_seg[batch_idx].unsqueeze(0).detach().cpu()
                    predict_seg = output_seg[batch_idx].unsqueeze(0).detach().cpu()

                    seg_prediction = torch.argmax(predict_seg, dim=1)
                    self.metric_evaluator_iou.update(seg_prediction, gt_seg_one)

            if self.train_detect:
                #---------------------------------------------------#
                #  检测部分metric
                #---------------------------------------------------#
                for idx in range(batch_size_tmp):
                    outs_detect = ([outputs["detection"][0][0][idx].unsqueeze(0)],
                                   [outputs["detection"][1][0][idx].unsqueeze(0)],
                                   [outputs["detection"][2][0][idx].unsqueeze(0)])
                    img_metas = batch_data["img_metas"][idx]
                    if self.use_distribute:
                        results_list = self.ultranet.module.center_head.get_bboxes(*outs_detect, img_metas, rescale=True)

                    else:
                        results_list = self.ultranet.center_head.get_bboxes(*outs_detect, img_metas, rescale=True)
                    bbox_result = [
                        bbox2result(det_bboxes, det_labels, self.num_class_detect)
                        for det_bboxes, det_labels in results_list
                    ]

                    detect_result.append(bbox_result[0])

            if self.train_lane:
                outs_lane = outputs["lane"]
                gt_lane_org_label = batch_data["gt_lane_org_label"]
                img_metas = batch_data["img_metas"]

                if self.use_distribute:
                    x,y = self.ultranet.module.lane_header.decode_result(outs_lane)

                else:
                    x,y = self.ultranet.lane_header.decode_result(outs_lane)


                for i, j,gt_lane_one_label,img_meta in zip(x, y,gt_lane_org_label,img_metas):
                    org_height,org_width,_ = img_meta[0]["ori_shape"]

                    ratio_w = self.net_input_width / float(org_width)
                    ratio_h = self.net_input_height / float(org_height)

                    temp_x, temp_y = convert_to_original_size(i, j, ratio_w, ratio_h)
                    temp_x, _ = fitting([temp_x], [temp_y], [gt_lane_one_label["h_samples"]], org_width)

                    if self.first_time:
                        self.gt_datas.append(gt_lane_one_label)

                    result_data_tmp_ = gt_lane_one_label.copy()
                    result_data_tmp_["lanes"] = temp_x[0]
                    result_data_tmp_["run_time"] = 1
                    lane_result.append(result_data_tmp_)


        # ---------------------------------------------------#
        #  分割部分metric summary
        # ---------------------------------------------------#
        if self.train_seg:
            print("=========================== metric segmentation %i ===========================" % epoch)
            scores = self.metric_evaluator_iou.compute()
            mIOU = list()
            for key, value in zip(self.segment_class_list, scores):
                print(key + ", " + "%.3f" % value)
                mIOU.append(value)
            mIOU = np.array(mIOU).mean()
            print("mIOU" + ", " + "%.3f" % mIOU)
            print()

        # ---------------------------------------------------#
        #  检测部分metric summary
        # ---------------------------------------------------#
        if self.train_detect:
            print("=========================== metric detection %i ===========================" % epoch)
            # 准备evaluator
            evaluator = CocoDataset(self.val_gt_json, self.class_list)
            metric = evaluator.evaluate_detect(detect_result, **self.eval_kwargs)
            print(metric)
            print()

        # ---------------------------------------------------#
        #  车道线部分metric summary
        # ---------------------------------------------------#
        if self.train_lane:
            print("=========================== metric lane %i ===========================" % epoch)
            if os.path.exists("test_result.json"):
                os.remove("test_result.json")
            save_result(lane_result, "test_result.json")

            if self.first_time:
                save_result(self.gt_datas,"test_label.json")
                self.first_time = False

            print(LaneEval.bench_one_submit("test_result.json", "test_label.json"))
            print()

        # 保存模型
        torch.save(self.ultranet.state_dict(), os.path.join(self.model_save_dir,"epoch_%i.pth" % epoch))
        return

def main(cfg_path):
    cfgs = yaml.safe_load(open(cfg_path))
    trainer = UltraTrainer(cfgs, cfg_path)
    epoch_all = cfgs["train"]["epoch"]
    fine_tuning = cfgs["train"]["fine_tuning"]
    backbone_lr_ratio_to_base = cfgs["train"]["backbone_lr_ratio_to_base"]

    if fine_tuning:
        epoch_tunning = cfgs["train"]["epoch_tuning"]
        tuning_turn = cfgs["train"]["tuning_turn"]
        assert (3 * epoch_tunning * tuning_turn <= epoch_all)
        epoch_joint = int(epoch_all / tuning_turn) - epoch_tunning * 3
        if trainer.use_distribute:
            group_backbone_neck = list(trainer.ultranet.module.backbone.parameters()) + list(
                trainer.ultranet.module.neck.parameters())
            group_lane = list(trainer.ultranet.module.lane_header.parameters())
            group_det = list(trainer.ultranet.module.center_head.parameters())
            group_seg = list(trainer.ultranet.module.seg_header.parameters())
        else:
            group_backbone_neck = list(trainer.ultranet.backbone.parameters()) + list(
                trainer.ultranet.neck.parameters())
            group_lane = list(trainer.ultranet.lane_header.parameters())
            group_det = list(trainer.ultranet.center_head.parameters())
            group_seg = list(trainer.ultranet.seg_header.parameters())

        group_back = trainer.optimizer.param_groups[0].copy()
        trainer.optimizer.param_groups.clear()
        trainer.optimizer.param_groups.append(group_back.copy())  # backbone + neck
        trainer.optimizer.param_groups.append(group_back.copy())  # lane
        trainer.optimizer.param_groups.append(group_back.copy())  # detection
        trainer.optimizer.param_groups.append(group_back.copy())  # segmentation

        # 赋值
        trainer.optimizer.param_groups[0]["params"] = group_backbone_neck
        trainer.optimizer.param_groups[1]["params"] = group_lane
        trainer.optimizer.param_groups[2]["params"] = group_det
        trainer.optimizer.param_groups[3]["params"] = group_seg

    else:
        epoch_joint = None
        epoch_tunning = None

    for epoch in range(epoch_all):
        # 是否单独tuning每一个分支头
        if fine_tuning:
            curr_turn = int(epoch / (epoch_joint + epoch_tunning * 3) )
            epoch_this_turn = epoch % (epoch_joint + epoch_tunning * 3)

            if epoch_this_turn < epoch_joint:
                print("======= TURN %i JOINT TRAINING =======" % curr_turn)
                if not trainer.flag_joint:
                    trainer.flag_joint = True
                    trainer.flag_det = False
                    lr_base = trainer.optimizer.param_groups[3]["lr"]
                    trainer.optimizer.param_groups[0]["lr"] = lr_base
                    trainer.optimizer.param_groups[1]["lr"] = lr_base
                    trainer.optimizer.param_groups[2]["lr"] = lr_base
                    trainer.optimizer.param_groups[3]["lr"] = lr_base

            # lane tuning
            elif epoch_joint <= epoch_this_turn < epoch_joint + epoch_tunning:
                print("======= TURN %i LANE TRAINING =======" % curr_turn)
                if not trainer.flag_lane:
                    trainer.flag_lane = True
                    trainer.flag_joint = False
                    lr_base = trainer.optimizer.param_groups[0]["lr"]
                    trainer.optimizer.param_groups[0]["lr"] = lr_base * backbone_lr_ratio_to_base
                    trainer.optimizer.param_groups[1]["lr"] = lr_base
                    trainer.optimizer.param_groups[2]["lr"] = 0.0
                    trainer.optimizer.param_groups[3]["lr"] = 0.0


            # det tuning
            elif epoch_joint + epoch_tunning <= epoch_this_turn < epoch_joint + 2 * epoch_tunning:
                print("======= TURN %i DET TRAINING =======" % curr_turn)
                if not trainer.flag_det:
                    trainer.flag_det = True
                    trainer.flag_lane = False
                    lr_base = trainer.optimizer.param_groups[1]["lr"]
                    trainer.optimizer.param_groups[0]["lr"] = lr_base * backbone_lr_ratio_to_base
                    trainer.optimizer.param_groups[1]["lr"] = 0.0
                    trainer.optimizer.param_groups[2]["lr"] = lr_base
                    trainer.optimizer.param_groups[3]["lr"] = 0.0

            # seg tuning
            else:
                print("======= TURN %i SEG TRAINING =======" % curr_turn)
                if not trainer.flag_seg:
                    trainer.flag_seg = True
                    trainer.flag_det = False
                    lr_base = trainer.optimizer.param_groups[2]["lr"]
                    trainer.optimizer.param_groups[0]["lr"] = lr_base * backbone_lr_ratio_to_base
                    trainer.optimizer.param_groups[1]["lr"] = 0.0
                    trainer.optimizer.param_groups[2]["lr"] = 0.0
                    trainer.optimizer.param_groups[3]["lr"] = lr_base

        trainer.train_one_epoch(epoch)
        print("=========================== VALIDATION %i ===========================" %epoch)
        trainer.valid(epoch)

    print("============== finish training ==============")
    return


def gpu_set(gpu_begin, gpu_number):
    gpu_id_str = ""
    for i in range(gpu_begin, gpu_number + gpu_begin):
        if i != gpu_begin + gpu_number - 1:
            gpu_id_str = gpu_id_str + str(i) + ","
        else:
            gpu_id_str = gpu_id_str + str(i)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    gpu_set(0, 4)

    # backbone mit b1
    # cfg_path = "cfgs/ultranet_mit_b1.yml"
    # cfg_path = "cfgs/ultranet_mit_b1_fine_tune.yml"

    # backbone resnet18
    # cfg_path = "cfgs/ultranet_resnet18.yml"
    # cfg_path = "cfgs/ultranet_resnet18_fine_tune.yml"

    # backbone resnet34
    # cfg_path = "cfgs/ultranet_resnet34.yml"
    cfg_path = "cfgs/ultranet_resnet34_fine_tune.yml"

    main(cfg_path)

    # 杀死所有进程号从2开始的进程
    # fuser -v /dev/nvidia* | grep 2* | xargs kill -9
