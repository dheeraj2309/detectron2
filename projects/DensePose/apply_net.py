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
    def get_output_path(cls: type, file_name: str, output_dir: str, new_ext: str, input_root: str = None):
        """Helper to determine the output path while preserving directory structure."""
        if input_root:
            # Ensure input_root is a directory path
            root = input_root if os.path.isdir(input_root) else os.path.dirname(input_root)
            relative_dir = os.path.relpath(os.path.dirname(file_name), root)
            out_dir = os.path.join(output_dir, relative_dir)
        else:
            out_dir = output_dir
        
        base_name = os.path.basename(file_name)
        out_name = os.path.splitext(base_name)[0] + new_ext
        out_path = os.path.join(out_dir, out_name)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return out_path
    
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
        """
        plain_result = {}
        # For Chart-based models (i, u, v)
        if hasattr(densepose_result, "labels") and densepose_result.labels is not None:
            # labels is the segmentation mask for the parts, equivalent to I
            plain_result["labels"] = densepose_result.labels.cpu()
        if hasattr(densepose_result, "uv") and densepose_result.uv is not None:
            plain_result["uv"] = densepose_result.uv.cpu()
        # For CSE-based models (embedding)
        if hasattr(densepose_result, "embedding") and densepose_result.embedding is not None:
            plain_result["embedding"] = densepose_result.embedding.cpu()
        return plain_result

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
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
            else:
                result["pred_densepose"] = []
                context["results"].append(result)
                return

            pred_densepose_results = extractor(outputs)[0]
            
            # Convert each custom object to a plain dictionary
            plain_densepose_results = [
                cls._convert_densepose_to_dict(dp_res) for dp_res in pred_densepose_results
            ]
            result["pred_densepose"] = plain_densepose_results

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
            logger.info(f"Plain data output saved to {out_fname}")
        logger.info("This file can now be loaded anywhere with PyTorch, without the DensePose project.")

@register_action
class DumpNpzAction(InferenceAction):
    """
    Dump action that outputs results to a compressed .npz file for each image.
    This is highly recommended for its small file size and portability.
    """
    COMMAND: ClassVar[str] = "dumpnpz"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(
            cls.COMMAND, help="Dump densepose data to a compressed .npz file per image."
        )
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpNpzAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<output_dir>",
            default="output/densepose_npz",
            help="Directory to save .npz files to. Preserves input folder structure.",
        )
        parser.add_argument(
            "--input-root",
            metavar="<input_root>",
            default=None,
            help="Root directory of the input dataset for relative path calculation.",
        )

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        return {"output_dir": args.output, "input_root": args.input_root}

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        file_name = entry["file_name"]
        out_path = cls.get_output_path(file_name, context["output_dir"], ".npz", context["input_root"])

        if not outputs.has("pred_densepose") or len(outputs) == 0:
            # Save an empty file to indicate no detection
            np.savez_compressed(out_path, scores=np.array([]))
            return

        # We are interested in chart-based outputs (I, U, V)
        if not isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
            logger.warning(f"Model output is not Chart-based for {file_name}. Skipping.")
            np.savez_compressed(out_path, scores=np.array([]))
            return
            
        extractor = DensePoseResultExtractor()
        results_densepose = extractor(outputs)[0]
        
        # Take the detection with the highest score
        dp_result = results_densepose[0]
        box_xywh = outputs.pred_boxes.tensor[0].cpu().numpy() # x,y,w,h
        score = outputs.scores[0].cpu().numpy()

        i_map = dp_result.labels.cpu().numpy()
        uv_map = dp_result.uv.cpu().numpy() # Shape: (2, H, W)
        
        np.savez_compressed(
            out_path,
            scores=score,
            pred_boxes_XYWH=box_xywh,
            pred_densepose_I=i_map,
            pred_densepose_UV=uv_map,
        )

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        logger.info(f"Finished processing. NPZ files saved in {context['output_dir']}")

@register_action
class IUVAction(InferenceAction):
    """
    Action to generate and save IUV-map images for a dataset.
    This version is robust to off-screen bounding boxes.
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
            default="output/densepose_iuv",
            help="Directory to save IUV image outputs.",
        )
        parser.add_argument(
            "--input-root",
            metavar="<input_root>",
            default=None,
            help="Root directory of the input dataset for relative path calculation.",
        )

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        return {"output_dir": args.output, "input_root": args.input_root}

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        file_name = entry["file_name"]
        out_path = cls.get_output_path(file_name, context["output_dir"], ".png", context["input_root"])

        img_h, img_w, _ = entry["image"].shape
        iuv_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        # Only proceed if there are detections with the correct densepose output type
        if (
            outputs.has("pred_densepose") and len(outputs) > 0 and 
            isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput)
        ):
            extractor = DensePoseResultExtractor()
            dp_result = extractor(outputs)[0][0] # Get the top-scoring result
            box_xyxy = outputs.pred_boxes.tensor[0].cpu().int().numpy()

            i_map_full = dp_result.labels.cpu().numpy().astype(np.uint8)
            uv_map_full = dp_result.uv.cpu().numpy()
            
            # BGR order for OpenCV: I -> Blue, U -> Green, V -> Red
            u_map = (uv_map_full[1, :, :] * 255).astype(np.uint8)
            v_map = (uv_map_full[0, :, :] * 255).astype(np.uint8)
            iuv_in_box = np.stack([i_map_full, u_map, v_map], axis=2)

            # --- ROBUST PASTING LOGIC TO PREVENT IndexError ---
            # 1. Get box coordinates and clip to image dimensions
            x1_box, y1_box, x2_box, y2_box = box_xyxy
            x1_img = max(x1_box, 0)
            y1_img = max(y1_box, 0)
            x2_img = min(x2_box, img_w)
            y2_img = min(y2_box, img_h)

            # 2. If the box is completely outside the image, do nothing
            if x1_img >= x2_img or y1_img >= y2_img:
                pass
            else:
                # 3. Calculate source coordinates from the unclipped box data
                x1_src = x1_img - x1_box
                y1_src = y1_img - y1_box
                x2_src = x2_img - x1_box
                y2_src = y2_img - y1_box
                
                # 4. Slice the data and mask from the source
                iuv_patch = iuv_in_box[y1_src:y2_src, x1_src:x2_src]
                mask = iuv_patch[:, :, 0] > 0 # Use the 'I' channel as the mask

                # 5. Get the destination view and paste using the mask
                dest_view = iuv_image[y1_img:y2_img, x1_img:x2_img]
                dest_view[mask] = iuv_patch[mask]

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
