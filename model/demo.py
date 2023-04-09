import yaml, os, time
import numpy as np
import cv2
np.random.seed(1991)

import torch.utils.data.dataloader
import warnings
warnings.filterwarnings("ignore")
from model import Ultranet

# detection
from head_detect.display import display_
from head_detect.centernet import  bbox2result
from torchvision.ops.boxes import batched_nms


def imagenet_normalize( img):
    """Normalize image.

    :param img: img that need to normalize
    :type img: RGB mode ndarray
    :return: normalized image
    :rtype: numpy.ndarray
    """
    pixel_value_range = np.array([255, 255, 255])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / pixel_value_range
    img = img - mean
    img = img / std
    return img


def deparallel_model(dict_param):
    ck_dict_new = dict()
    for key, value in dict_param.items():
        temp_list = key.split(".")[1:]
        new_key = ""
        for tmp in temp_list:
            new_key += tmp + "."
        ck_dict_new[new_key[0:-1]] = value
    return ck_dict_new


def decode(imgs, masks, org_size, vis_color_id):
    seg_predictions = torch.argmax(masks, dim=1).detach().cpu().numpy()

    batch_size = len(imgs)
    visual_imgs = list()
    for batch_idx in range(batch_size):
        seg_prediction = seg_predictions[batch_idx]
        im_vis = imgs[batch_idx]

        # vis
        vis_seg = np.zeros([seg_prediction.shape[0], seg_prediction.shape[1], 3], dtype=np.uint8)
        for cls_id, color in vis_color_id.items():
            vis_seg[seg_prediction == cls_id] = color
        vis_seg = cv2.resize(vis_seg, org_size, cv2.INTER_NEAREST)
        im_vis = cv2.addWeighted(im_vis, 0.8, vis_seg, 0.5, 0.0)
        visual_imgs.append(im_vis)

    return visual_imgs


if __name__ == '__main__':
    deploy = False
    img_test = True
    display = False
    use_fix_color = True
    #---------------------------------------------------#
    #  参数设定
    #---------------------------------------------------#
    log_path = "logs/joint_resnet34_finetune"
    model_name = "epoch_13.pth"

    img_folder = "demo/images"
    video_path = "demo/video/test_video.avi"

    # 导出参数
    cfg_path = os.path.join(log_path, "config.yml")
    cfgs = yaml.safe_load(open(cfg_path))
    net_input_width = cfgs["dataloader"]["network_input_width"]
    net_input_height = cfgs["dataloader"]["network_input_height"]
    net_input_size = (net_input_width, net_input_height)
    train_detect = cfgs["train"]["train_detect"]
    train_seg = cfgs["train"]["train_seg"]
    train_lane = cfgs["train"]["train_lane"]
    use_distribute = cfgs["train"]["use_distribute"]

    # 设定分割相关
    segment_class_list = cfgs["segment"]["class_list"]
    # seg_class_color_id = {0: (0, 0, 0),
    #                       1: (128, 0, 128),
    #                       2: (255, 255, 255),
    #                       3: (0, 255, 255),
    #                       4:(0,255,0)
    #                       }
    seg_class_color_id = {0: (0, 0, 0),
                          1: (128, 0, 128),
                          2: (255, 255, 255),
                          }

    if not use_fix_color:
        seg_class_color_id = dict()
        for idx in range(len(segment_class_list)):
            color = (
                int(np.random.randint(128, 255)), int(np.random.randint(128, 255)), int(np.random.randint(128, 255)))
            seg_class_color_id.update({idx: color})

    # 设定加载检测相关
    conf_threshold_detect = 0.3
    iou_threshold = 0.4
    obj_list = cfgs["detection"]["class_list"][1:]

    #---------------------------------------------------#
    #  网络模型
    #---------------------------------------------------#
    if deploy:
        ultranet = Ultranet(cfgs=cfgs,onnx_export=True).cuda()
    else:
        ultranet = Ultranet(cfgs=cfgs,onnx_export=False).cuda()

    dict_old = torch.load(os.path.join(log_path,"model",model_name))
    if use_distribute:
        dict_new = deparallel_model(dict_old)
    else:
        dict_new = dict_old

    ultranet.load_state_dict(dict_new)
    ultranet.eval()

    #---------------------------------------------------#
    #  导出模型onnx
    #---------------------------------------------------#
    if deploy:
        import torch.onnx
        output_list = ["output_seg", "topk_scores", "topk_inds", "wh_pred", "offset_pred", "out_confidence", "out_offset","out_instance"]
        dummy_input = torch.randn([1, 3, 640, 640]).to("cuda:0")
        torch.onnx.export(
            ultranet,(dummy_input, "deploy"),
            "ultraNET.onnx",
            export_params=True,
            input_names=["input", "mode"],
            output_names=output_list,
            opset_version=12,
            verbose=False,
        )

        exit()

    #---------------------------------------------------#
    #  测试模型
    #---------------------------------------------------#
    if img_test:
        img_list = os.listdir(img_folder)
        if not os.path.exists("demo/images_vis"): os.makedirs("demo/images_vis")
        vid = None
        video_writer = None
    else:
        vid = cv2.VideoCapture(video_path)
        video_output = video_path.replace("video","video_vis")
        if not os.path.exists("demo/video_vis"): os.makedirs("demo/video_vis")
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(video_output, codec, 10, (1920,1080))
        img_list = None

        if display:
            cv2.namedWindow("visual",cv2.WINDOW_FREERATIO)

    counter = 0
    while True:

        if img_test:
            if counter > len(img_list) - 1:
                break

            img_name = img_list[counter]
            img_path = os.path.join("demo/images",img_name)
            input_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            print("process %s" % img_name)
        else:
            _, input_img = vid.read()
            img_path = None

        if input_img is None:
            break
        counter+=1

        org_height, org_width = input_img.shape[0:2]
        org_size = (org_width , org_height)

        #---------------------------------------------------#
        #  preprocess
        #---------------------------------------------------#
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, net_input_size)
        img = img.astype(np.float32)
        img = imagenet_normalize(img)
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        img = torch.tensor(img).cuda().float()

        #---------------------------------------------------#
        #  inference
        #---------------------------------------------------#
        tic = time.time()
        outputs = ultranet(img)
        print("inference time is %i" %(1000*(time.time() - tic)))

        #---------------------------------------------------#
        #  decoder
        #---------------------------------------------------#
        imgs = [input_img]
        if train_lane:
            lane_outs = outputs["lane"]
            out_x_, out_y_ = ultranet.lane_header.decode_result(lane_outs)
            ratio_w = net_input_width / float(org_width)
            ratio_h = net_input_height / float(org_height)

            images = ultranet.lane_header.display(out_x_, out_y_, imgs, ratio_w, ratio_h)


        if train_seg:
            # display
            output_seg = outputs["seg"]
            imgs = decode(imgs, output_seg, org_size, seg_class_color_id)


        if train_detect:
            target_size = (org_width, org_height)
            num_classes = cfgs["detection"]["num_classes"]
            detect_outs = outputs["detection"]
            img_meta = dict()
            img_meta["ori_shape"] = (org_height, org_width, 3)
            img_meta["img_shape"] = (net_input_height, net_input_width, 3)
            img_meta["pad_shape"] = (net_input_height, net_input_width, 3)
            img_meta["scale_factor"] = [1., 1., 1., 1.]
            img_meta["flip"] = False
            img_meta["flip_direction"] = None
            img_meta["border"] = [0, org_height, 0, org_width]
            img_meta["batch_input_shape"] = (org_height, org_width)
            img_metas = [img_meta]
            results_list = ultranet.center_head.get_bboxes(*detect_outs, img_metas, rescale=True)
            bbox_results = [bbox2result(det_bboxes, det_labels, num_classes)
                            for det_bboxes, det_labels in results_list][0]

            # filter bboxes
            filter_list = list()
            class_index = list()
            score_list = list()
            for i in range(num_classes):
                bbox_this_class = bbox_results[i]
                size_pred = bbox_this_class.shape[0]
                for j in range(size_pred):
                    conf_ = bbox_this_class[j,-1]
                    if conf_ > conf_threshold_detect:
                        filter_list.append(bbox_this_class[j,0:-1])
                        class_index.append(i)
                        score_list.append(conf_)

            # nms
            filter_list = torch.tensor(filter_list)
            score_list = torch.tensor(score_list)
            class_index = torch.tensor(class_index)
            anchors_nms_idx = batched_nms(filter_list, score_list, class_index,
                                          iou_threshold=iou_threshold)

            score_list = score_list[anchors_nms_idx]
            filter_list = filter_list[anchors_nms_idx]
            class_index = class_index[anchors_nms_idx]
            imgs = display_(imgs, filter_list, class_index,score_list, obj_list)


        print("total process time is %i" %(1000*(time.time() - tic)))

        # ---------------------------------------------------#
        #  保存显示
        # ---------------------------------------------------#
        if not img_test:
            if display:
                cv2.imshow('visual', imgs[0])
                cv2.waitKey(1)
            video_writer.write(imgs[0])

            # if counter > 500:
            #     break

        else:
            cv2.imwrite(img_path.replace("images", "images_vis"), cv2.resize(imgs[0],(720,400)))

