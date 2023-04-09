from abc import ABC
import warnings

import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from mit.net import resize

def _build_head(in_channel, feat_channel, out_channel):
    """Build head for each branch."""
    layer = nn.Sequential(
        nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(feat_channel, out_channel, kernel_size=1))
    return layer





class SegHeader(nn.Module, ABC):
    def __init__(self,cfgs):
        super(SegHeader,self).__init__()
        self.cfgs = cfgs
        self.net_input_height = self.cfgs["dataloader"]["network_input_height"]
        self.net_input_width = self.cfgs["dataloader"]["network_input_width"]
        self.in_channel = self.cfgs["backbone"]["channels"]
        self.feat_channel = self.cfgs["segment"]["feat_channel"]
        self.out_channel = len(self.cfgs["segment"]["class_list"])
        self.align_corners = self.cfgs["backbone"]["align_corners"]
        self.seg_header = _build_head(self.in_channel,self.feat_channel,self.out_channel)

    def forward(self, x):
        output_seg_up_one = resize(
            input_=x,
            size=(x.shape[2] * 2, x.shape[3] * 2),
            mode='bilinear',
            align_corners=self.align_corners
        )

        output_seg_up_one = self.seg_header(output_seg_up_one)
        output_seg = resize(
            input_=output_seg_up_one,
            size=(self.net_input_height, self.net_input_width),
            mode='bilinear',
            align_corners=self.align_corners
        )

        return output_seg

if __name__ == '__main__':
    import yaml
    CFG_PATH = "../cfgs/ultranet_tencent6524.yml"
    cfgs = yaml.safe_load(open(CFG_PATH))
    seg_header = SegHeader(cfgs).cuda()
    dummy_input = torch.randn((2, 256, 80, 80)).cuda()
    out_seg = seg_header(dummy_input)
    print(out_seg.shape)