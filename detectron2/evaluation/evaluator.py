# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

import pdb


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                            k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
        model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], metadata=None, visualize=False, vis_thr=0.5
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
            #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
            #     log_every_n_seconds(
            #         logging.INFO,
            #         (
            #             f"Inference done {idx + 1}/{total}. "
            #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
            #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
            #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
            #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
            #             f"ETA={eta}"
            #         ),
            #         n=5,
            #     )
            start_data_time = time.perf_counter()

            # visualize outputs, and instances of inputs

            # Assuming `inputs` is your input image batch and `outputs` are the outputs from your model
            if visualize:
                for input, output in zip(inputs, outputs):
                    # Get the original image (before any data augmentation or preprocessing)
                    original_image = input["image"].cpu().numpy().transpose(1, 2, 0)

                    img_h, img_w = original_image.shape[:2]
                    pred_h, pred_w = output["instances"].image_size

                    original_image = cv2.resize(original_image, (pred_w, pred_h))
                    # original_image = (original_image * std + mean).astype("uint8")

                    # Create a Visualizer instance
                    v = Visualizer(original_image, metadata)

                    # draw original gt objects
                    gt_boxes = input["instances"].gt_boxes.tensor.cpu().numpy()
                    gt_boxes[:, 0::2] *= pred_w / img_w  # Scale x coordinates
                    gt_boxes[:, 1::2] *= pred_h / img_h
                    gt_names = [metadata.thing_classes[c] for c in input["instances"].gt_classes.cpu().numpy()]
                    gt_colors = ["green"] * len(gt_names)

                    # draw synthetic objects
                    synth_boxes = input["instances_synth"].gt_boxes.tensor.cpu().numpy()
                    synth_boxes[:, 0::2] *= pred_w / img_w  # Scale x coordinates
                    synth_boxes[:, 1::2] *= pred_h / img_h
                    synth_names = [metadata.thing_classes[c] + "-synth" for c in input["instances_synth"].gt_classes.cpu().numpy()]
                    synth_colors = ["red"] * len(synth_names)

                    # draw prediction
                    conf_idx = output["instances"].scores > vis_thr
                    boxes = output["instances"][conf_idx].pred_boxes.tensor.cpu().numpy()
                    pred_boxes = boxes.copy()
                    # pred_boxes[:, 0::2] *= img_w / pred_w  # Scale x coordinates
                    # pred_boxes[:, 1::2] *= img_h / pred_h
                    classes = output["instances"][conf_idx].pred_classes.cpu().numpy()
                    scores = output["instances"][conf_idx].scores.cpu().numpy()
                    pred_names = ["{} {:.1f}%".format(metadata.thing_classes[c], s * 100) for c, s in zip(classes, scores)]
                    pred_colors = ["blue"]*len(pred_names)

                    boxes = np.concatenate([gt_boxes, synth_boxes, pred_boxes], axis=0)
                    class_names = gt_names + synth_names + pred_names
                    colors = gt_colors + synth_colors + pred_colors
                    # boxes = np.concatenate([gt_boxes, pred_boxes], axis=0)
                    # class_names = gt_names  + pred_names
                    # colors = gt_colors  + pred_colors
                    v = v.overlay_instances(boxes=boxes, labels=class_names, assigned_colors=colors)

                    # Use the `draw_instance_predictions` method to overlay the predictions on the image
                    # v = v.draw_instance_predictions(output["instances"].to("cpu"))

                    # Convert the image from BGR to RGB (this is required by matplotlib)
                    image_rgb = cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2RGB)
                    # Display the image
                    # fig, ax = plt.subplots()
                    # ax.imshow(image_rgb)
                    # ax.set_aspect('equal')  # Set aspect ratio to 'equal' to remove empty space
                    # plt.show()
                    # Display the image
                    plt.imshow(image_rgb)
                    plt.savefig(os.path.join('vis/foggy_synthetic', input['file_name'].split('/')[-1]), dpi=300)

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )

    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def analyze_on_dataset(
        model, data_loader, class_names
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.\

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    class_names = class_names + ['bg'] + [c + '_synth' for c in class_names]

    features = [torch.Tensor([]).to(model.device) for _ in class_names]
    predictions = [torch.Tensor([]).to(model.device) for _ in class_names]
    # preds_cnt = torch.zeros((len(class_names), model.roi_heads.num_classes + 1), dtype=torch.long).to(model.device)

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        for idx, inputs in enumerate(data_loader):
            if idx % 100 == 0:
                print(idx)
            if idx > 1000:
                break
            box_features, preds, gt_classes = model.extract_features(inputs)
            for i, cls in enumerate(class_names):
                features[i] = torch.cat([features[i], box_features[gt_classes == i]], dim=0)
                predictions[i] = torch.cat([predictions[i], preds[0][gt_classes == i].argmax(dim=1)])
                # if sum(gt_classes == i) > 0:
                #     preds_cnt[i] += torch.bincount(predictions[0][gt_classes == i].softmax(dim=1).argmax(dim=1))
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    from sklearn.manifold import TSNE
    import numpy as np
    import matplotlib.pyplot as plt

    # Randomly sample up to 200 features for each class
    sampled_features = []
    labels = []
    preds = []
    for i, feat in enumerate(features):
        if feat.size(0) > 1000:
            indices = torch.randperm(feat.size(0))[:1000]
        else:
            indices = torch.arange(feat.size(0))
        sampled_features.append(feat[indices])
        labels += [i] * indices.size(0)  # Store class index for each feature
        preds.append(predictions[i][indices])

    # Concatenate all sampled features into a single tensor and convert labels to a tensor
    all_features = torch.cat([f.mean(dim=[2, 3]) for f in sampled_features], dim=0).cpu().numpy()
    labels = np.array(labels)
    all_preds = torch.cat(preds).cpu().numpy()

    # Use t-SNE to reduce dimensions to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    # Plot the 2D features with different colors for each class
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # colors_gt = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    colors_gt = [
                    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
                    "#f032e6", "#bcf60c"] * 2
    # color_list = [
    #     "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    #     "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8",
    #     "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"
    # ]
    # for i, class_name in enumerate(class_names):
    for i, class_name in enumerate(class_names):
        # if 'synth' not in class_name:
        #     continue
        indices = labels == i
        mark = "o" if "synth" in class_name else "x"
        axs[0].scatter(features_2d[indices, 0], features_2d[indices, 1], label=class_name, color=colors_gt[i],
                       marker=mark)

    axs[0].set_title('t-SNE visualization by actual classes')
    axs[0].set_xlabel('t-SNE feature 1')
    axs[0].set_ylabel('t-SNE feature 2')
    axs[0].legend()

    # Plot for predicted classes
    # for i, class_name in enumerate(['car', 'bg']):
    for i, class_name in enumerate(class_names[:9]):
        # plot real objects
        indices = (all_preds == i) & (labels < 8)
        axs[1].scatter(features_2d[indices, 0], features_2d[indices, 1], label=class_name, color=colors_gt[i],
                       marker='x')
        # plot synthetic objects
        indices = (all_preds == i) & (labels > 8)
        axs[1].scatter(features_2d[indices, 0], features_2d[indices, 1], label=class_name + '_synth',
                       color=colors_gt[i], marker='o')
    axs[1].set_title('t-SNE visualization by prediction classes')
    axs[1].set_xlabel('t-SNE feature 1')
    axs[1].set_ylabel('t-SNE feature 2')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('t-SNE_of_foggy_synthetic_inpainting_describe_0327_via_cs_model')


# import matplotlib.pyplot as plt
#
# # Assuming `features_2d` is your 2D feature set from t-SNE,
# # `labels` are your actual class labels,
# # and `prediction_classes` are your predicted class labels
#
# # Convert labels and prediction_classes to numpy arrays for consistency
# labels = np.array(labels)
# prediction_classes = np.array(prediction_classes)
#
# # Generate a unique list of actual and predicted classes for legends and coloring
# unique_actual_classes = np.unique(labels)
# unique_predicted_classes = np.unique(prediction_classes)
# colors_actual = plt.cm.rainbow(np.linspace(0, 1, len(unique_actual_classes)))
# colors_predicted = plt.cm.rainbow(np.linspace(0, 1, len(unique_predicted_classes)))
#
# fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # 1 row, 2 columns for two plots
#
# # Plot for actual classes
# for i, class_name in enumerate(class_names):  # Assuming class_names contains the names of actual classes
#     indices = labels == i
#     axs[0].scatter(features_2d[indices, 0], features_2d[indices, 1], color=colors_actual[i], label=class_name)
# axs[0].set_title('t-SNE visualization of actual classes')
# axs[0].set_xlabel('t-SNE feature 1')
# axs[0].set_ylabel('t-SNE feature 2')
# axs[0].legend()
#
# # Plot for predicted classes
# for i, class_name in enumerate(unique_predicted_classes):
#     indices = prediction_classes == class_name
#     axs[1].scatter(features_2d[indices, 0], features_2d[indices, 1], color=colors_predicted[i], label=class_name)
# axs[1].set_title('t-SNE visualization colored by prediction classes')
# axs[1].set_xlabel('t-SNE feature 1')
# axs[1].set_ylabel('t-SNE feature 2')
# axs[1].legend()
#
# plt.tight_layout()
# plt.show()


def inference_on_corruption_dataset(
        model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            base_dict = {k: v for k, v in inputs[0].items() if "image" not in k}
            base_dict["image_id"] = inputs[0]["image_id"]
            for severity in range(4, 5):
                corrupt_inputs = base_dict.copy()
                corrupt_inputs["image"] = inputs[0]["image_" + str(severity)]
                corrupt_inputs = [corrupt_inputs]
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(corrupt_inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(corrupt_inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                #     log_every_n_seconds(
                #         logging.INFO,
                #         (
                #             f"Inference done {idx + 1}/{total}. "
                #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
                #             f"ETA={eta}"
                #         ),
                #         n=5,
                #     )
                start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )

    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
