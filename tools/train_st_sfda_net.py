#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import copy
import time
import datetime
import wandb
import torch.optim as optim
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    ClipartDetectionEvaluator,
    WatercolorDetectionEvaluator,
    CityscapeDetectionEvaluator,
    FoggyDetectionEvaluator,
    CityscapeCarDetectionEvaluator,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

import pdb
import cv2
from pynvml import *
from detectron2.structures.boxes import Boxes, pairwise_iou
from detectron2.structures.instances import Instances
from detectron2.data.detection_utils import convert_image_to_rgb

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if evaluator_type == "clipart":
        return ClipartDetectionEvaluator(dataset_name)
    if evaluator_type == "watercolor":
        return WatercolorDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape":
        return CityscapeDetectionEvaluator(dataset_name)
    if evaluator_type == "foggy":
        return FoggyDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape_car":
        return CityscapeCarDetectionEvaluator(dataset_name)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

# =====================================================
# ================== Pseduo-labeling ==================
# =====================================================
def threshold_bbox(proposal_bbox_inst, thres=0.7, proposal_type="roih", synth_inst=None, iou_thres=0.7):
    if proposal_type == "rpn":
        valid_map = proposal_bbox_inst.objectness_logits > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
            valid_map
        ]
    elif proposal_type == "roih":
        valid_map = proposal_bbox_inst.scores > thres

        # if synthetic objects exist,

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        if synth_inst is not None:
            synth_inst = synth_inst.to(new_boxes.device)
            # add synthetic object and boxes and pseudo labeled boxes to instances
            iou = pairwise_iou(new_boxes, synth_inst.gt_boxes)
            non_overlapped = iou.max(dim=1)[0] < iou_thres
            new_proposal_inst.gt_boxes = Boxes.cat([new_boxes[non_overlapped], synth_inst.gt_boxes])

            pseudo_labeled_classes = proposal_bbox_inst.pred_classes[valid_map][non_overlapped]
            new_proposal_inst.gt_classes = torch.cat([pseudo_labeled_classes, synth_inst.gt_classes])

            pseudo_labeled_scores = proposal_bbox_inst.scores[valid_map][non_overlapped]
            synth_scores = torch.ones(len(synth_inst)).to(new_boxes.device)
            new_proposal_inst.scores = torch.cat([pseudo_labeled_scores, synth_scores])
        else:
            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

    return new_proposal_inst


def process_pseudo_label(proposals_rpn_k, cur_threshold, proposal_type, psedo_label_method="", synth_instances=None):
    list_instances = []
    num_proposal_output = 0.0
    for idx, proposal_bbox_inst in enumerate(proposals_rpn_k):
        # thresholding
        if psedo_label_method == "thresholding":
            synth_inst = synth_instances[idx] if synth_instances is not None else None
            proposal_bbox_inst = threshold_bbox(
                proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type, synth_inst=synth_inst
            )
        else:
            raise ValueError("Unkown pseudo label boxes methods")
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)
    num_proposal_output = num_proposal_output / len(proposals_rpn_k)
    return list_instances, num_proposal_output

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996, except_backbone=False, first_update=False):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model_student.state_dict().items()
        }
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            if except_backbone and "backbone" in key:
                new_teacher_dict[key] = value
            else:
                if first_update:
                    new_teacher_dict[key] = student_model_dict[key]
                else:
                    new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                    )
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict


def update_feature_bank(feature_bank, features, predictions=None, gt_classes=None, thr=0.5, max_num=100, device='cuda'):
    scores, classes = torch.nn.Softmax(dim=1)(predictions[0]).max(dim=1) if predictions is not None else (torch.ones_like(gt_classes).to(device), gt_classes.to(device))
    for cls in torch.unique(classes):
        cur_feats = features[(scores > thr) & (classes == cls)].mean(dim=[2,3]).detach()
        cur_feats /= torch.norm(cur_feats, dim=1, keepdim=True)
        feature_bank[cls] = torch.cat([feature_bank[cls], cur_feats], dim=0)
        feature_bank[cls] = feature_bank[cls][-max_num:]


def filter_out_noise_feature(features, instances, original_feature_bank, synthetic_feature_bank, thr=0.5):
    if features is None:
        return None
    normalized_features = features.mean(dim=[2,3]) / torch.norm(features.mean(dim=[2,3]), dim=1, keepdim=True)
    similarity_with_original = torch.zeros((features.size(0), len(original_feature_bank))).to(features.device)
    similarity_with_synthetic = torch.zeros((features.size(0), len(synthetic_feature_bank))).to(features.device)
    for cls, feats in enumerate(original_feature_bank):
        if feats.size(0) == 0:
            continue
        cosine_similarity = torch.mm(normalized_features, feats.t())
        max_similarity, _ = cosine_similarity.max(dim=1)
        similarity_with_original[:, cls] = max_similarity
    max_similarity_original, max_cls_original = similarity_with_original.max(dim=1)

    # for cls, feats in enumerate(synthetic_feature_bank):
    #     if feats.size(0) == 0:
    #         continue
    #     cosine_similarity = torch.mm(normalized_features, feats.t())
    #     max_similarity, _ = cosine_similarity.max(dim=1)
    #     similarity_with_synthetic[:, cls] = max_similarity
    # max_similarity_synthetic, max_cls_synthetic = similarity_with_synthetic.max(dim=1)

    gt_classes = [inst.gt_classes for inst in instances if inst is not None]
    idx = 0
    filtered_instances = []
    for i, inst in enumerate(instances):
        if inst is not None:
            # gt_classes = inst.gt_classes.to(max_cls_original.device)
            valid1 = ((inst.gt_classes == 0) | (inst.gt_classes == 2)) & (inst.gt_classes == max_cls_original[idx:idx + len(inst.gt_classes)].cpu())
            valid2 = (inst.gt_classes != 0) & (inst.gt_classes != 2)
            filtered_instances.append(inst[valid1 | valid2])
            idx += len(inst.gt_classes)
        else:
            filtered_instances.append(None)

    return filtered_instances


def visualize_proposals(cfg, batched_inputs, proposals, box_size, proposal_dir, metadata):
        from detectron2.utils.visualizer import Visualizer

        for input, prop in zip(batched_inputs, proposals):
            img = input["image_weak"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), None)
            #v_gt = Visualizer(img, None)
            #v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            #anno_img = v_gt.get_image()
            v_pred = Visualizer(img, metadata)
            if proposal_dir == "rpn":
                v_pred = v_pred.overlay_instances( boxes=prop.proposal_boxes[0:int(box_size)].tensor.cpu().numpy())
            if proposal_dir == "roih":
                v_pred = v_pred.draw_instance_predictions(prop)
            vis_img = v_pred.get_image()

            save_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir) 
            save_img_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir, input['file_name'].split('/')[-1]) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_img_path, vis_img)


def visualize_instances(cfg, batched_inputs, pseudo_labels, synth_instances, metadata, thr=0.9, vis_dir='vis'):
    from detectron2.utils.visualizer import Visualizer

    for input, pseudo, synth in zip(batched_inputs, pseudo_labels, synth_instances):
        img = input["image_weak"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), None)
        # v_gt = Visualizer(img, None)
        # v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
        # anno_img = v_gt.get_image()
        v_pred = Visualizer(img, metadata)
        v_pred = v_pred.draw_instance_pred_synth(pseudo[pseudo.scores > thr], synth)
        vis_img = v_pred.get_image()

        save_path = os.path.join(cfg.OUTPUT_DIR, vis_dir)
        save_img_path = os.path.join(cfg.OUTPUT_DIR, vis_dir, input['file_name'].split('/')[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_img_path, vis_img)


def test_sfda(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        cfg.defrost()
        cfg.SOURCE_FREE.TYPE = False
        cfg.freeze()
        test_data_loader = build_detection_test_loader(cfg, dataset_name)
        test_metadata = MetadataCatalog.get(dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, test_data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            #pdb.set_trace()
            cls_names = test_metadata.get("thing_classes")
            cls_aps = results_i['bbox']['class-AP50']
            for i in range(len(cls_aps)):
                logger.info("AP for {}: {}".format(cls_names[i], cls_aps[i]))
    if len(results) == 1:
        results = list(results.values())[0]
    return results, cls_names, cls_aps


def train_sfda(cfg, model_student, model_teacher, resume=False):
    
    checkpoint = copy.deepcopy(model_teacher.state_dict())

    model_teacher.eval()
    model_student.train()

    #optimizer = optim.SGD(model_student.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = build_optimizer(cfg, model_student)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model_student, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    #pdb.set_trace()

    data_loader = build_detection_train_loader(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    total_epochs = cfg.SOLVER.TOTAL_EPOCH
    len_data_loader = len(data_loader.dataset.dataset.dataset) // cfg.SOLVER.IMS_PER_BATCH
    start_iter, max_iter = 0, len_data_loader
    max_sf_da_iter = total_epochs*max_iter
    logger.info("Starting training from iteration {}".format(start_iter))

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, len_data_loader, max_iter=max_sf_da_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_sf_da_iter) if comm.is_main_process() else []

    model_teacher.eval()
    first_update = True
    # results, cls_names, cls_aps = test_sfda(cfg, model_teacher)
    # wandb_log = {"teacher-{}".format(name): ap for name, ap in zip(cls_names, cls_aps)}
    # wandb_log["teacher-AP"] = results['bbox']['AP']
    # wandb_log["teacher-AP50"] = results['bbox']['AP50']
    # wandb.log(wandb_log, step=0)

    original_feature_bank = [torch.Tensor([]).to(model_teacher.device) for _ in range(len(metadata.thing_classes)+1)]
    synthetic_feature_bank = [torch.Tensor([]).to(model_teacher.device) for _ in range(len(metadata.thing_classes)+1)]

    start_time = time.perf_counter()
    iters_after_start = 0
    with EventStorage(start_iter) as storage:
        for epoch in range(1, total_epochs+1):
            cfg.defrost()
            cfg.SOURCE_FREE.TYPE = True
            cfg.freeze()
            data_loader = build_detection_train_loader(cfg)
            model_teacher.eval()
            model_student.train()
            if cfg.ADAPT.ONLY_HEAD:
                model_student.backbone.requires_grad_(False)
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                storage.iter = iteration
                iters_after_start += 1
                optimizer.zero_grad()

                with torch.no_grad():
                    _, teacher_features, teacher_proposals, teacher_results = model_teacher(data, mode="train")

                # teacher_pseudo_proposals, num_rpn_proposal = process_pseudo_label(teacher_proposals, 0.9, "rpn", "thresholding")
                # teacher_pseudo_results, num_roih_proposal = process_pseudo_label(teacher_results, 0.9, "roih", "thresholding")

                synth_instances = [d["instances_synth"] if "instances_synth" in d else None for d in data]

                pred_features, pred_predictions = model_teacher.roi_heads.forward_with_gt_boxes(teacher_features, teacher_results)
                synth_features, synth_predictions = model_teacher.roi_heads.forward_with_gt_boxes(teacher_features, synth_instances)

                update_feature_bank(original_feature_bank, pred_features, predictions=pred_predictions, thr=0.5, max_num=100, device=model_teacher.device)

                filtered_synth_instances = filter_out_noise_feature(synth_features, synth_instances, original_feature_bank, synthetic_feature_bank, thr=0.5)
                # synth_gt_classes = [inst.gt_classes for inst in synth_instances if inst is not None]
                # if len(synth_gt_classes) > 0:
                #     # update_feature_bank(synthetic_feature_bank, synth_features, gt_classes=torch.cat(synth_gt_classes), thr=0.5, max_num=100)
                #     update_feature_bank(synthetic_feature_bank, synth_features, filtered_synth_instances, gt_classes=torch.cat(synth_gt_classes), thr=0.5, max_num=100)
                # teacher_pseudo_results, num_roih_proposal = process_pseudo_label(teacher_results, cfg.ADAPT.PSEUDO_THRESH, "roih", "thresholding", synth_instances=synth_instances)
                teacher_pseudo_results, num_roih_proposal = process_pseudo_label(teacher_results, cfg.ADAPT.PSEUDO_THRESH, "roih", "thresholding", synth_instances=filtered_synth_instances)

                loss_dict = model_student(data, cfg, model_teacher, teacher_features, teacher_proposals, teacher_pseudo_results, mode="train")

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                # visualize_instances(cfg, data, teacher_results, synth_instances, metadata, vis_dir='vis')

                if iteration - start_iter > 5 and ((iteration + 1) % 50 == 0 or iteration == max_iter - 1):
                    total_seconds_per_batch = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_batch * (total_epochs * max_iter - iters_after_start)))
                    print("Epoch {}/{} Iter {}/{}".format(epoch, total_epochs, iteration, max_iter), "lr:", optimizer.param_groups[0]["lr"], ''.join(['{}: {:.3f}, '.format(k, v.item()) for k,v in loss_dict.items()]), "ETA:", eta)

                # wandb log
                wandb_log = {k: v.item() for k, v in loss_dict.items()}
                wandb_log["lr"] = optimizer.param_groups[0]["lr"]
                wandb.log(wandb_log, step=iters_after_start)

                periodic_checkpointer.step(iteration)

                if iters_after_start % cfg.ADAPT.EMA_PERIOD == 0:
                    new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate=cfg.ADAPT.EMA_RATIO, except_backbone=cfg.ADAPT.ONLY_HEAD, first_update=first_update)
                    first_update = False
                    model_teacher.load_state_dict(new_teacher_dict)
                    model_teacher.eval()
                    print("Teacher model testing@", epoch)
                    results, cls_names, cls_aps = test_sfda(cfg, model_teacher)
                    wandb_log = {"teacher-{}".format(name): ap for name, ap in zip(cls_names, cls_aps)}
                    wandb_log["teacher-AP"] = results['bbox']['AP']
                    wandb_log["teacher-AP50"] = results['bbox']['AP50']
                    wandb.log(wandb_log, step=iters_after_start)

            # save checkpoint and evaluate every epoch
            model_student.eval()
            print("Student model testing@", epoch)
            results, cls_names, cls_aps = test_sfda(cfg, model_student)
            wandb_log = {"student-{}".format(name): ap for name, ap in zip(cls_names, cls_aps)}
            wandb_log["student-AP"] = results['bbox']['AP']
            wandb_log["student-AP50"] = results['bbox']['AP50']
            wandb.log(wandb_log, step=iters_after_start)

            torch.save(model_teacher.state_dict(), cfg.OUTPUT_DIR + "/model_teacher_{}.pth".format(epoch))
            torch.save(model_student.state_dict(), cfg.OUTPUT_DIR + "/model_student_{}.pth".format(epoch))
    
    model_student.eval()
    print("Student model testing@", epoch)
    test_sfda(cfg, model_student)

    model_teacher.eval()
    print("Teacher model testing@", epoch)
    test_sfda(cfg, model_teacher)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    cfg = setup(args)

    model_student = build_model(cfg)
    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = "teacher_sfda_RCNN"
    cfg.freeze()
    model_teacher = build_model(cfg)
    logger.info("Model:\n{}".format(model_student))

    DetectionCheckpointer(model_student, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)

    # modify roi head to adapt to the additional classes
    if cfg.ADAPT.CLS:
        model_student.roi_heads.box_predictor.extend_classes(cfg.ADAPT.CLS_MATCH)
        model_teacher.roi_heads.box_predictor.extend_classes(cfg.ADAPT.CLS_MATCH)
        model_student.to(torch.device(cfg.MODEL.DEVICE))
        model_teacher.to(torch.device(cfg.MODEL.DEVICE))

    wandb.init(project="Open-OD-NeurIPS2024")
    wandb.run.name = cfg.OUTPUT_DIR

    logger.info("Trained model has been successfully loaded")
    return train_sfda(cfg, model_student, model_teacher)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
