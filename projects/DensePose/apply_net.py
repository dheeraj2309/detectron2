#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import logging
import os
import sys
from typing import Any, ClassVar, Dict, List
import torch
import numpy as np
from tqdm import tqdm

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)

DOC = """Apply Net - a tool to print / visualize DensePose results
"""

LOGGER_NAME = "apply_net"
logger = logging.getLogger(LOGGER_NAME)

_ACTION_REGISTRY: Dict[str, "Action"] = {}


class Action:
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.",
        )


def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class InferenceAction(Action):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(InferenceAction, cls).add_arguments(parser)
        parser.add_argument("cfg", metavar="<config>", help="Config file")
        parser.add_argument("model", metavar="<model>", help="Model file")
        parser.add_argument("input", metavar="<input>", help="Input data")
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )

    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        logger.info(f"Loading config from {args.cfg}")
        opts = []
        cfg = cls.setup_config(args.cfg, args.model, args, opts)
        logger.info(f"Loading model from {args.model}")
        predictor = DefaultPredictor(cfg)
        logger.info(f"Loading data from {args.input}")
        file_list = cls._get_input_file_list(args.input)
        if len(file_list) == 0:
            logger.warning(f"No input images for {args.input}")
            return
        # Create context and add total number of entries
        context = cls.create_context(args, cfg)
        context["total_entries"] = len(file_list)

        for i, file_name in enumerate(file_list):
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            # Add current entry index to context
            context["entry_idx"] = i
            with torch.no_grad():
                outputs = predictor(img)["instances"]
                cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
        cls.postexecute(context)

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list
    
    # NEW a_get_out_fname METHOD IN THE BASE CLASS
    @classmethod
    def _get_out_fname(cls: type, context: Dict[str, Any], entry_idx: int, fname_base: str):
        """
        Gets the output file name for a given entry index.
        If there is only one entry, returns the base file name.
        Otherwise, appends a formatted index to the file name.
        """
        base, ext = os.path.splitext(fname_base)
        if context["total_entries"] > 1:
            return base + ".{0:04d}".format(entry_idx) + ext
        return fname_base
    
    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        logger.info(f"Loading config from {args.cfg}")
        opts = []
        cfg = cls.setup_config(args.cfg, args.model, args, opts)
        logger.info(f"Loading model from {args.model}")
        predictor = DefaultPredictor(cfg)
        logger.info(f"Loading data from {args.input}")
        file_list = cls._get_input_file_list(args.input)
        if len(file_list) == 0:
            logger.warning(f"No input images for {args.input}")
            return
        # Create context and add total number of entries
        context = cls.create_context(args, cfg)
        context["total_entries"] = len(file_list)

        # ------------------- PROGRESS BAR CHANGE IS HERE -------------------
        # Wrap the file_list with tqdm to create a progress bar
        print(f"Found {len(file_list)} images to process.")
        iterable = tqdm(file_list, desc="Processing Images")
        
        for i, file_name in enumerate(iterable):
        # -------------------------------------------------------------------
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            # Add current entry index to context
            context["entry_idx"] = i
            with torch.no_grad():
                outputs = predictor(img)["instances"]
                cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
        cls.postexecute(context)


@register_action
class DumpAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file
    """

    COMMAND: ClassVar[str] = "dump"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Dump model outputs to a file.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="results.pkl",
            help="File name to save dump to",
        )

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
        context["results"].append(result)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        context = {"results": [], "out_fname": args.output}
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "wb") as hFile:
            torch.save(context["results"], hFile)
            logger.info(f"Output saved to {out_fname}")

@register_action
class DumpPlainAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file using only plain data
    (tensors, dicts, lists) to avoid dependency issues on loading.
    """

    COMMAND: ClassVar[str] = "dumpplain"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(
            cls.COMMAND, help="Dump model outputs to a file using plain data types."
        )
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpPlainAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="results_plain.pkl",
            help="File name to save plain data dump to",
        )

    @classmethod
    def _convert_densepose_to_dict(cls: type, densepose_result):
        """
        Converts a DensePoseData object to a dictionary of tensors.
        This is the core of the solution.
        """
        # DensePoseData is a dataclass, we can inspect its fields.
        # Common fields include 'uv', 'labels', 'i', 's'. We save them as tensors.
        # We use .cpu() to move data from GPU to CPU before saving.
        plain_result = {}
        if hasattr(densepose_result, "uv"):
            plain_result["uv"] = densepose_result.uv.cpu()
        if hasattr(densepose_result, "labels"):
            # labels is the segmentation mask for the parts
            plain_result["labels"] = densepose_result.labels.cpu()
        # For Chart-based models (i, u, v)
        if hasattr(densepose_result, "i"):
            plain_result["i"] = densepose_result.i.cpu()
        # For CSE-based models (embedding)
        if hasattr(densepose_result, "embedding"):
            plain_result["embedding"] = densepose_result.embedding.cpu()

        return plain_result

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        result = {"file_name": image_fpath}

        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()

        if outputs.has("pred_densepose"):
            # Determine the correct extractor based on the output type
            if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                extractor = DensePoseResultExtractor()
            elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                extractor = DensePoseOutputsExtractor()
            else:
                # If we don't know the type, we can't extract safely.
                result["pred_densepose"] = []
                context["results"].append(result)
                return

            # pred_densepose_results is a list of custom DensePoseData objects
            pred_densepose_results = extractor(outputs)[0]
            
            # This is the crucial part: convert each custom object to a plain dictionary
            plain_densepose_results = [
                cls._convert_densepose_to_dict(dp_res) for dp_res in pred_densepose_results
            ]
            result["pred_densepose"] = plain_densepose_results

        context["results"].append(result)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        # We don't need the verification code from the original postexecute
        context = {"results": [], "out_fname": args.output}
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "wb") as hFile:
            torch.save(context["results"], hFile)
            logger.info(f"Output saved to {out_fname}")
        logger.info("Successfully saved plain data. This file can now be loaded anywhere without the DensePose project.")

@register_action
class IUVAction(InferenceAction):
    """
    Action that outputs IUV images.
    An IUV image is a 3-channel image where the channels are encoded as follows:
    - Channel 0 (Blue): V coordinate
    - Channel 1 (Green): U coordinate
    - Channel 2 (Red): Body part index (I)
    Note: OpenCV uses BGR order, so we save (V, U, I) to get (B, G, R).
    """

    COMMAND: ClassVar[str] = "iuv"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Dump IUV image outputs.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(IUVAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="output_iuv.png",
            help="File name to save IUV image to.",
        )

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        import cv2
        import numpy as np # Ensure numpy is imported

        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")

        # Ensure model output is for chart-based models which have I,U,V data
        if not outputs.has("pred_densepose") or not isinstance(
            outputs.pred_densepose, DensePoseChartPredictorOutput
        ):
            logger.warning(
                f"Could not find DensePoseChartPredictorOutput in outputs for {image_fpath}. Skipping."
            )
            return

        # Create a blank (black) image to draw the IUV data on
        h, w, _ = entry["image"].shape
        iuv_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Extractor gets the IUV data from the raw model output
        extractor = DensePoseResultExtractor()
        densepose_results = extractor(outputs)[0]

        if not densepose_results:
            logger.warning(f"No DensePose detections for {image_fpath}. Saving black image.")
        else:
            # Iterate over each detected instance
            for i, result in enumerate(densepose_results):
                x, y, w_box, h_box = outputs.pred_boxes.tensor[i].int().tolist()
                
                i_map = result.labels.squeeze().to(torch.uint8)
                u_map = (result.uv.squeeze()[0] * 255).to(torch.uint8)
                v_map = (result.uv.squeeze()[1] * 255).to(torch.uint8)

                # OpenCV uses BGR order, so we stack (V, U, I) to save correctly.
                iuv_patch = torch.stack([v_map, u_map, i_map], dim=-1).cpu().numpy()
                mask = iuv_patch[:, :, 2] > 0
                
                # Ensure the patch fits within the image boundaries
                y1, y2 = y, y + h_box
                x1, x2 = x, x + w_box
                patch_h, patch_w, _ = iuv_patch.shape
                
                # Clip coordinates to be within image bounds
                y1_p, y2_p = max(0, y1), min(h, y2)
                x1_p, x2_p = max(0, x1), min(w, x2)

                # Calculate corresponding slice from the patch
                y1_patch = y1_p - y1
                y2_patch = y1_patch + (y2_p - y1_p)
                x1_patch = x1_p - x1
                x2_patch = x1_patch + (x2_p - x1_p)

                # Paste the valid part of the IUV patch onto the main IUV image
                iuv_image[y1_p:y2_p, x1_p:x2_p][mask[y1_patch:y2_patch, x1_patch:x2_patch]] = \
                    iuv_patch[y1_patch:y2_patch, x1_patch:x2_patch][mask[y1_patch:y2_patch, x1_patch:x2_patch]]

        entry_idx = context["entry_idx"]
        out_fname = cls._get_out_fname(context, entry_idx, context["out_fname"])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        cv2.imwrite(out_fname, iuv_image)
        logger.info(f"IUV image saved to {out_fname}")

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode) -> Dict[str, Any]:
        context = { "out_fname": args.output }
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

@register_action
class BatchIUVAction(InferenceAction):
    """
    Action that outputs IUV images using batch processing for higher throughput on GPU.
    """

    COMMAND: ClassVar[str] = "batchiuv"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(
            cls.COMMAND, help="Dump IUV images in batches for high performance."
        )
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(BatchIUVAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="output_iuv.png",
            help="File name template to save IUV images to.",
        )
        parser.add_argument(
            "--batch-size",
            metavar="<N>",
            type=int,
            default=8,
            help="Number of images to process in a single batch. Adjust based on GPU memory.",
        )


    @staticmethod
    def _create_iuv_image_from_output(original_image, outputs):
        """Helper function to generate a single IUV image from a model output."""
        import cv2
        import numpy as np

        h, w, _ = original_image.shape
        iuv_image = np.zeros((h, w, 3), dtype=np.uint8)

        if not outputs.has("pred_densepose") or not isinstance(
            outputs.pred_densepose, DensePoseChartPredictorOutput
        ):
            print("DEBUG: No DensePoseChartPredictorOutput found. Returning black image.")
            return iuv_image

        extractor = DensePoseResultExtractor()
        densepose_results = extractor(outputs)[0]

        if not densepose_results:
            print("DEBUG: No DensePose detections in this image. Returning black image.")
            return iuv_image

        print(f"\n--- DEBUG: Processing an image with {len(densepose_results)} detected instances ---")
        for i, result in enumerate(densepose_results):
            print(f"\n--- Instance {i+1}/{len(densepose_results)} ---")
            
            # --- START OF DEBUGGING ---
            x1, y1, x2, y2 = outputs.pred_boxes.tensor[i].int().tolist()
            print(f"DEBUG: BBox (x1,y1,x2,y2): ({x1}, {y1}, {x2}, {y2})")

            # Let's inspect the raw result tensors
            print(f"DEBUG: Raw result.labels shape: {result.labels.shape}")
            print(f"DEBUG: Raw result.uv shape: {result.uv.shape}")

            i_map_raw = result.labels
            uv_map_raw = result.uv

            # Squeeze to remove any singleton dimensions
            i_map = i_map_raw.squeeze()
            uv_map = uv_map_raw.squeeze()
            
            print(f"DEBUG: Squeezed i_map shape: {i_map.shape} (should be 2D)")
            print(f"DEBUG: Squeezed uv_map shape: {uv_map.shape} (should be 3D: 2 x H x W)")

            # Defensive check
            if i_map.dim() != 2 or uv_map.dim() != 3 or uv_map.shape[0] != 2:
                print(f"ERROR-DEBUG: Unexpected map dimensions! Skipping this instance.")
                continue

            u_map = uv_map[0, :, :]
            v_map = uv_map[1, :, :]
            print(f"DEBUG: Extracted u_map shape: {u_map.shape}")
            print(f"DEBUG: Extracted v_map shape: {v_map.shape}")
            
            # This is the line that creates the 3-channel torch tensor
            try:
                iuv_patch_torch = torch.stack([v_map, u_map, i_map], dim=-1)
                print(f"DEBUG: torch.stack result `iuv_patch_torch` shape: {iuv_patch_torch.shape} (should be 3D: H x W x 3)")
            except Exception as e:
                print(f"ERROR-DEBUG: torch.stack failed! Error: {e}")
                continue

            # Convert to numpy
            iuv_patch = iuv_patch_torch.cpu().numpy()
            print(f"DEBUG: Converted to numpy `iuv_patch` shape: {iuv_patch.shape} (should be 3D: H x W x 3)")

            # Check if resizing is needed, which was a potential point of failure
            w_box, h_box = x2 - x1, y2 - y1
            patch_h, patch_w, patch_c = iuv_patch.shape
            print(f"DEBUG: BBox size (W, H): ({w_box}, {h_box}) vs Patch size (W, H): ({patch_w}, {patch_h})")
            
            if h_box != patch_h or w_box != patch_w:
                print(f"DEBUG: Resizing needed. Resizing from ({patch_w}, {patch_h}) to ({w_box}, {h_box})")
                try:
                    iuv_patch = cv2.resize(iuv_patch, (w_box, h_box), interpolation=cv2.INTER_NEAREST)
                    print(f"DEBUG: Post-resize `iuv_patch` shape: {iuv_patch.shape} (should be 3D)")
                except Exception as e:
                    print(f"ERROR-DEBUG: cv2.resize failed! Error: {e}")
                    continue
            
            # This is the line that failed before. Let's check the shape right before it.
            print(f"DEBUG: FINAL `iuv_patch` shape before masking: {iuv_patch.shape}")
            if iuv_patch.ndim != 3:
                 print(f"CRITICAL-DEBUG: `iuv_patch` IS NOT 3D! It has {iuv_patch.ndim} dimensions. This will fail.")

            # THE CRASHING LINE
            mask = iuv_patch[:, :, 2] > 0
            # --- END OF DEBUGGING ---

            # Clip coordinates to be safely within the image boundaries
            y1_c, y2_c = max(0, y1), min(h, y2)
            x1_c, x2_c = max(0, x1), min(w, x2)
            
            y_patch_start, y_patch_end = y1_c - y1, y2_c - y1
            x_patch_start, x_patch_end = x1_c - x1, x2_c - x1
            
            valid_patch = iuv_patch[y_patch_start:y_patch_end, x_patch_start:x_patch_end]
            valid_mask = mask[y_patch_start:y_patch_end, x_patch_start:x_patch_end]
            
            region_to_write = iuv_image[y1_c:y2_c, x1_c:x2_c]
            
            # Final check to prevent broadcast errors
            if region_to_write[valid_mask].shape == valid_patch[valid_mask].shape:
                region_to_write[valid_mask] = valid_patch[valid_mask]
            else:
                print("ERROR-DEBUG: Shape mismatch during final paste. Skipping paste for this instance.")
        
        return iuv_image

    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        import cv2
        from detectron2.data import transforms as T

        logger.info(f"Loading config from {args.cfg}")
        cfg = cls.setup_config(args.cfg, args.model, args, [])
        logger.info(f"Loading model from {args.model}")
        
        # We need the model directly, not the DefaultPredictor
        model = DefaultPredictor(cfg).model
        
        logger.info(f"Loading data from {args.input}")
        file_list = cls._get_input_file_list(args.input)
        if len(file_list) == 0:
            logger.warning(f"No input images for {args.input}")
            return
            
        # Create batches of file names
        batch_size = args.batch_size
        batches = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]
        
        file_idx = 0
        context = {"total_entries": len(file_list), "out_fname": args.output}
        
        for batch_files in tqdm(batches, desc=f"Processing in batches of {batch_size}"):
            # 1. Load and preprocess a batch of images
            batch_inputs = []
            original_images = []
            for file_name in batch_files:
                original_image = read_image(file_name, format="BGR")
                original_images.append(original_image)
                height, width = original_image.shape[:2]
                # Preprocessing from DefaultPredictor logic
                aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
                image = aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                batch_inputs.append({"image": image, "height": height, "width": width})

            # 2. Run inference on the whole batch
            with torch.no_grad():
                # The model can take a list of inputs and processes them in a batch
                batch_outputs = model(batch_inputs)
            
            # 3. Post-process and save results for this batch
            for i in range(len(batch_outputs)):
                outputs_per_image = {"instances": batch_outputs[i]["instances"]}
                iuv_image = cls._create_iuv_image_from_output(original_images[i], outputs_per_image["instances"])
                
                if iuv_image is not None:
                    out_fname = cls._get_out_fname(context, file_idx, args.output)
                    out_dir = os.path.dirname(out_fname)
                    if len(out_dir) > 0 and not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    cv2.imwrite(out_fname, iuv_image)
                
                file_idx += 1

@register_action
class ShowAction(InferenceAction):
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Visualize selected entries")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument(
            "visualizations",
            metavar="<visualizations>",
            help="Comma separated list of visualizations, possible values: "
            "[{}]".format(",".join(sorted(cls.VISUALIZERS.keys()))),
        )
        parser.add_argument(
            "--min_score",
            metavar="<score>",
            default=0.8,
            type=float,
            help="Minimum detection score to visualize",
        )
        parser.add_argument(
            "--nms_thresh", metavar="<threshold>", default=None, type=float, help="NMS threshold"
        )
        parser.add_argument(
            "--texture_atlas",
            metavar="<texture_atlas>",
            default=None,
            help="Texture atlas file (for IUV texture transfer)",
        )
        parser.add_argument(
            "--texture_atlases_map",
            metavar="<texture_atlases_map>",
            default=None,
            help="JSON string of a dict containing texture atlas files for each mesh",
        )
        parser.add_argument(
            "--output",
            metavar="<image_file>",
            default="outputres.png",
            help="File name to save output to",
        )

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(args.min_score))
        if args.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(args.nms_thresh))
        cfg = super(ShowAction, cls).setup_config(config_fpath, model_fpath, args, opts)
        return cfg

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        import cv2
        import numpy as np

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"]
        # UPDATED CALL to _get_out_fname
        out_fname = cls._get_out_fname(context, entry_idx, context["out_fname"])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_fname, image_vis)
        logger.info(f"Output saved to {out_fname}")

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass


    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": args.output,
            # entry_idx is now managed by the base execute method
        }
        return context


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=120),
    )
    parser.set_defaults(func=lambda _: parser.print_help(sys.stdout))
    subparsers = parser.add_subparsers(title="Actions")
    for _, action in _ACTION_REGISTRY.items():
        action.add_parser(subparsers)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    verbosity = getattr(args, "verbosity", None)
    global logger
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(verbosity_to_level(verbosity))
    args.func(args)


if __name__ == "__main__":
    main()
