import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from inference_thumos import inference
from utils import misc_utils
from torch.utils.data import Dataset
from dataset.thumos_features import ThumosFeature
from model_factory import ModelFactory
from utils.loss import CrossEntropyLoss, GeneralizedCE
from config.config_thumos import Config, parse_args, class_dict

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "CAS_Only.pkl")
        print("loading from file for training: ", model_file)
        pretrained_params = torch.load(model_file)

        selected_params = OrderedDict()
        for k, v in pretrained_params.items():
            if 'base_module' in k:
                selected_params[k] = v

        model_dict = net.state_dict()
        model_dict.update(selected_params)
        net.load_state_dict(model_dict)


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


def train(net, config):
    # resume training
    load_weight(net, config)

    # set up dataloader, loss and optimizer
    train_loader, test_loader = get_dataloaders(config)
    ce_criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr,
                                 betas=(0.9, 0.999), weight_decay=0.0005)

    # training
    best_mAP = -1
    step = 0
    cumloss = 0
    for epoch in range(config.num_epochs):

        for _data, _label, _, _, _ in train_loader:

            batch_size = _data.shape[0]
            _data, _label = _data.cuda(), _label.cuda()
            optimizer.zero_grad()

            # FORWARD PASS
            cas, (action_flow, action_rgb) = net(_data, is_training=True)

            combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1), 
                                action_flow.permute(0, 2, 1).detach(), action_rgb.permute(0, 2, 1))
            _, topk_indices = torch.topk(combined_cas, config.num_segments // 8, dim=1)

            cas_top = torch.gather(cas, 1, topk_indices)
            cas_top = torch.mean(cas_top, dim=1)

            # calcualte pseudo target
            cls_agnostic_gt = []
            cls_agnostic_neg_gt = []
            for b in range(batch_size):
                label_indices_b = torch.nonzero(_label[b, :])[:,0]
                topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
                cls_agnostic_gt_b = torch.zeros((1, 1, config.num_segments)).cuda()

                # positive examples
                for gt_i in range(len(label_indices_b)):
                    cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
                cls_agnostic_gt.append(cls_agnostic_gt_b)

            cls_agnostic_gt = torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments

            # losses
            base_loss = ce_criterion(cas_top, _label)
            base_loss_np = base_loss.detach().cpu().numpy()
            cost = base_loss

            pos_factor = torch.sum(cls_agnostic_gt,dim=2)
            neg_factor = torch.sum(1-cls_agnostic_gt,dim=2)

            Lgce = GeneralizedCE(q=config.q_val)

            cls_agnostic_loss_flow = Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1))
            cls_agnostic_loss_rgb = Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

            cost += cls_agnostic_loss_flow + cls_agnostic_loss_rgb
            cost.backward()

            optimizer.step()

            cumloss += cost.cpu().item()
            step += 1

            # evaluation
            if step % config.detection_inf_step == 0:
                cumloss /= config.detection_inf_step

                with torch.no_grad():
                    net = net.eval()
                    mean_ap, test_acc = inference(net, config, test_loader, model_file=None)
                    net = net.train()

                if mean_ap > best_mAP:
                    best_mAP = mean_ap
                    torch.save(net.state_dict(), os.path.join(config.model_path, "CAS_Only.pkl"))

                print("epoch={:5d}  step={:5d}  Loss={:.4f}  cls_acc={:5.2f}  best_map={:5.2f}".format(
                        epoch, step, cumloss, test_acc * 100, best_mAP * 100))

                cumloss = 0


def test(net, config):
    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    net.eval()

    with torch.no_grad():
        model_filename = "CAS_Only.pkl"
        writer = None
        config.model_file = os.path.join(config.model_path, model_filename)
        _mean_ap, test_acc = inference(net, config, test_loader, model_file=config.model_file)
        print("cls_acc={:.5f} map={:.5f}".format(test_acc*100, _mean_ap*100))


def main():
    args = parse_args()
    config = Config(args)

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False

    net = ModelFactory.get_model(config.model_name, config)
    net = net.cuda()

    if args.inference_only:
        test(net, config)
    else:
        train(net, config)


if __name__ == '__main__':
    main()
