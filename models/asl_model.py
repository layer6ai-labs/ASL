import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import math

class ASLModel(nn.Module):
    def __init__(self, len_feature, num_classes, num_segments, pooling_type="", config=None):
        super(ASLModel, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.config = config

        self.base_module = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0),
        )

        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0),
        )


    def forward(self, x, is_training, reweight_cas=False, do_supp=True, kmult=6, pool=False):
        input = x.permute(0, 2, 1)

        cas_fg = self.base_module(input).permute(0, 2, 1)
        action_flow = torch.sigmoid(self.action_module_flow(input[:, 1024:, :]))
        action_rgb = torch.sigmoid(self.action_module_rgb(input[:, :1024, :]))

        return cas_fg, (action_flow, action_rgb)
