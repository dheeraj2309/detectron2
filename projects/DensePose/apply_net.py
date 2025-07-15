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
import cv2

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
    Action to generate and save IUV-map images for a dataset.
    """

    COMMAND: ClassVar[str] = "iuv"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Generate IUV images for a dataset.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(IUVAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<output_dir>",
            default="output/densepose",
            help="Directory to save IUV image outputs. Will preserve input folder structure.",
        )
        # Add an argument to define the root of the input dataset for relative path calculation
        parser.add_argument(
            "--input-root",
            metavar="<input_root>",
            default=None,
            help="Root directory of the input dataset. If provided, paths will be relative to this root.",
        )

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        # The extractor to get DensePose results from model outputs
        extractor = DensePoseResultExtractor()
        context = {"extractor": extractor, "output_dir": args.output, "input_root": args.input_root}
        if args.input_root and not os.path.isdir(args.input_root):
            logger.error(f"Provided --input-root '{args.input_root}' is not a valid directory.")
            raise NotADirectoryError(f"Provided --input-root '{args.input_root}' is not a valid directory.")
        return context

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        extractor = context["extractor"]
        # Extract DensePose results from model outputs
        results_densepose = extractor(outputs)[0]
        pred_boxes = outputs.pred_boxes.tensor.cpu()

        # Create a blank image to store the IUV map
        h, w, _ = entry["image"].shape
        iuv_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Results are sorted by score, so later instances (higher score) overwrite earlier ones.
        if results_densepose:
            # result is a DensePoseData object
            dp_result = results_densepose[0]
            box = pred_boxes[0].to(dtype=torch.int).numpy()
            i_map_tensor = dp_result.labels
            uv_map_tensor = dp_result.uv

            i_map = i_map_tensor.cpu().numpy()
            # The UV tensor is shape (2, H, W). Index 1 is U, Index 0 is V.
            u_map = uv_map_tensor.cpu().numpy()[1, :, :]
            v_map = uv_map_tensor.cpu().numpy()[0, :, :]
            
            # Use BGR order for OpenCV: I -> Blue, U -> Green, V -> Red
            iuv_in_box = np.stack(
                [
                    i_map.astype(np.uint8),
                    (u_map * 255).astype(np.uint8),
                    (v_map * 255).astype(np.uint8),
                ],
                axis=2,
            )
            
            # Create a mask for where to paint the IUV data
            mask = i_map > 0
            x1,y1,x2,y2 = box
            
            # Define slices for copying data to avoid out-of-bounds errors
            img_y1_clipped = max(y1, 0)
            img_y2_clipped = min(y2, h)
            img_x1_clipped = max(x1, 0)
            img_x2_clipped = min(x2, w)
            
            # If the intersection is empty (e.g., box is completely outside image), do nothing.
            if img_y1_clipped >= img_y2_clipped or img_x1_clipped >= img_x2_clipped:
                pass
            else:
                # Create slices for the destination iuv_image
                img_slice_y = slice(img_y1_clipped, img_y2_clipped)
                img_slice_x = slice(img_x1_clipped, img_x2_clipped)
                
                # Create corresponding slices for the source iuv_in_box and the mask.
                # These are adjusted by the original box's top-left corner (x1, y1).
                box_slice_y = slice(img_y1_clipped - y1, img_y2_clipped - y1)
                box_slice_x = slice(img_x1_clipped - x1, img_x2_clipped - x1)
                
                # Now, the mask and the data we select from it will have the exact same dimensions
                # as the destination slice in the iuv_image.
                mask = i_map[box_slice_y, box_slice_x] > 0
                
                iuv_image[img_slice_y, img_slice_x][mask] = iuv_in_box[box_slice_y, box_slice_x][mask]
        # --- Determine and create output path ---
        file_name = entry["file_name"]
        input_root = context.get("input_root")

        if input_root:
            # Preserve directory structure relative to the input_root
            relative_path = os.path.relpath(file_name, input_root)
            out_path = os.path.join(context["output_dir"], relative_path)
        else:
            # Use a flat structure in the output directory
            base_name = os.path.basename(file_name)
            out_path = os.path.join(context["output_dir"], base_name)
            
        # Change extension to .png for lossless saving
        out_path = os.path.splitext(out_path)[0] + ".png"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, iuv_image)

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        logger.info(f"Finished processing. IUV images saved in {context['output_dir']}")

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
