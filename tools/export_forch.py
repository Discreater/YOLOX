import argparse
import json
import pickle
from typing import Dict, Any

import numpy as np
import torch
from loguru import logger
from torch import nn

import deploy.models.yolox
import yolox.exp
from yolox import models
from yolox import utils
from yolox.models import network_blocks as blocks

@logger.catch
def main():
    args = parse_args()
    logger.info("args value: {}".format(args))
    exp = yolox.exp.get_exp(args.exp_file, None)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.eval()

    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = utils.fuse_model(model)
    serialized_model = serialize_model(model)
    forch_model, state = exp.to_forch(model)
    compare(forch_model, model)
    with open("yolox.forch.state.pkl", "wb") as f:
        pickle.dump(state, f)
    with open(args.output_name, "wb") as f:
        pickle.dump(forch_model, f)

    remove_ndarray(serialized_model)

    if args.json_dst is not None:
        with open(args.json_dst, "w") as f:
            json.dump(serialized_model, f, indent=2)

def compare(fmodel: deploy.models.yolox.YOLOX , tmodel: yolox.models.YOLOX):
    image = torch.randn(1, 3, 320, 320)
    image_np = image.data.numpy().astype(np.float32)
    tr0 = tmodel.backbone(image)
    fr0 = fmodel.backbone(image_np)
    tr1 = tmodel.head(tr0)
    fr1 = fmodel.head(fr0)
    diff = np.abs(tr1.data.numpy() - fr1)

    print((diff < 0.1).all())

def serialize_model(m: nn.Module) -> Dict[str, Any]:
    if isinstance(m, models.YOLOX):
        return {
            "type": "YOLOX",
            "backbone": serialize_model(m.backbone),
            "head": serialize_model(m.head)
        }
    elif isinstance(m, models.YOLOXHead):
        return {
            "type": "YOLOXHead",
            "cls_convs": [serialize_model(sm) for sm in m.cls_convs],
            "reg_convs": [serialize_model(sm) for sm in m.reg_convs],
            "cls_preds": [serialize_model(sm) for sm in m.cls_preds],
            "reg_preds": [serialize_model(sm) for sm in m.reg_preds],
            "obj_preds": [serialize_model(sm) for sm in m.obj_preds],
            "stems": [serialize_model(sm) for sm in m.stems],
        }
    elif isinstance(m, models.YOLOPAFPN):
        return {
            "type": "YOLOPAFPN",
            "backbone": serialize_model(m.backbone),
            "upsample": serialize_model(m.upsample),
            "lateral_conv0": serialize_model(m.lateral_conv0),
            "C3_p4": serialize_model(m.C3_p4),
            "reduce_conv1": serialize_model(m.reduce_conv1),
            "C3_p3": serialize_model(m.C3_p3),
            "bu_conv2": serialize_model(m.bu_conv2),
            "C3_n3": serialize_model(m.C3_n3),
            "bu_conv1": serialize_model(m.bu_conv1),
            "C3_n4": serialize_model(m.C3_n4),
        }
    elif isinstance(m, models.darknet.CSPDarknet):
        return {
            "type": "CSPDarknet",
            "stem": serialize_model(m.stem),
            "dark2": serialize_model(m.dark2),
            "dark3": serialize_model(m.dark3),
            "dark4": serialize_model(m.dark4),
            "dark5": serialize_model(m.dark5),
        }
    elif isinstance(m, blocks.Focus):
        return {
            "type": "Focus",
            "conv": serialize_model(m.conv),
        }
    elif isinstance(m, blocks.BaseConv):
        return {
            "type": "BaseConv",
            "weight": m.conv.weight.data.cpu().numpy(),
            "bias": m.conv.bias.data.cpu().numpy(),
            "act": serialize_model(m.act),
        }
    elif isinstance(m, nn.Sequential):
        return {
            "type": "Sequential",
            "seq": [serialize_model(sm) for sm in m]
        }
    elif isinstance(m, blocks.DWConv):
        return {
            "type": "DWConv",
            "dconv": serialize_model(m.dconv),
            "pconv": serialize_model(m.pconv),
        }
    elif isinstance(m, blocks.SPPBottleneck):
        return {
            "type": "SPPBottleneck",
            "conv1": serialize_model(m.conv1),
            "m": [serialize_model(sm) for sm in m.m],
            "conv2": serialize_model(m.conv2),
        }
    elif isinstance(m, nn.SiLU):
        return {
            "type": "SiLU",
        }
    elif isinstance(m, blocks.CSPLayer):
        return {
            "type": "CSPLayer",
            "conv1": serialize_model(m.conv1),
            "conv2": serialize_model(m.conv2),
            "conv3": serialize_model(m.conv3),
            "m": serialize_model(m.m)
        }
    elif isinstance(m, blocks.Bottleneck):
        return {
            "type": "Bottleneck",
            "conv1": serialize_model(m.conv1),
            "conv2": serialize_model(m.conv2),
        }
    elif isinstance(m, nn.MaxPool2d):
        return {
            "type": "MaxPool2d",
        }
    elif isinstance(m, nn.Upsample):
        return {
            "type": "Upsample"
        }
    elif isinstance(m, nn.Conv2d):
        return {
            "type": "Conv2d",
            "weight": m.weight.data.cpu().numpy(),
            "bias": m.bias.data.cpu().numpy(),
        }
    else:
        logger.error("{}".format(type(m)))
        assert False, "unknown type"


def remove_ndarray(d: dict):
    for (k, v) in d.items():
        if isinstance(v, dict):
            remove_ndarray(v)
        elif isinstance(v, list):
            [remove_ndarray(var) for var in v if isinstance(var, dict)]
        elif type(v) is np.ndarray:
            d[k] = k


def parse_args():
    parser = argparse.ArgumentParser("YOLOX forch deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.forch.pkl", help="output name of models"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-c", "--ckpt", required=True, type=str, help="ckpt path")
    parser.add_argument("--experiment-name", default=None, type=str, help="experiment name of model")
    parser.add_argument("--json-dst", default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    main()
