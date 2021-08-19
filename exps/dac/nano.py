#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = "dac_nano"
        self.enable_mixup = False
        self.eval_interval = 1

        # Define yourself dataset path
        self.data_dir = "datasets/dac"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.num_classes = 1

        self.in_channels = [256, 512, 1024]

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=self.in_channels, depthwise=True, act="relu")
            head = YOLOXHead(self.num_classes, self.width, in_channels=self.in_channels, depthwise=True, act="relu")
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def to_forch(self, m: nn.Module):
        import deploy.models.blocks
        import deploy.models.yolox
        import numpy as np

        backbone = deploy.models.yolox.YOLOPAFPN(self.depth, self.width, in_channels=self.in_channels, depthwise=True)
        head = deploy.models.yolox.YOLOXHead(self.num_classes, self.width, in_channels=self.in_channels, depthwise=True)
        model = deploy.models.yolox.YOLOX(backbone, head)
        model.load_state_dict(m.state_dict())
        model.to_type(np.float32)
        state_dict = {
            "state": model.state_dict(),
            "depth": self.depth,
            "width": self.width,
            "in_channels": self.in_channels,
            "depthwise": True,
            "num_classes": self.num_classes
        }
        return model, state_dict
