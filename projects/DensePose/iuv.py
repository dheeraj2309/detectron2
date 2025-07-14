import argparse
import os
import cv2
import torch
import numpy as np

# --- Detectron2 / DensePose Imports ---
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
from densepose.structures import DensePoseResultExtractor
from densepose.vis.densepose_results import DensePoseResultsVisualizerWithTexture

def setup_densepose_predictor(config_fpath, model_fpath):
    """
    Sets up a DensePose predictor from a configuration file and a model file.
    
    Args:
        config_fpath (str): Path to the DensePose config file (.yaml).
        model_fpath (str): Path to the DensePose model weights file (.pkl).
        
    Returns:
        DefaultPredictor: An object that can be called to run inference.
    """
    print(f"Loading config from {config_fpath}")
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    
    print(f"Loading model from {model_fpath}")
    cfg.MODEL.WEIGHTS = model_fpath
    
    # Set device (GPU or CPU)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return predictor

def main():
    """
    Main function to run the DensePose IUV map generation.
    """
    parser = argparse.ArgumentParser(description="Generate a DensePose IUV map from a single image.")
    parser.add_argument(
        "--config",
        metavar="<config>",
        required=True,
        help="Path to the DensePose config file (e.g., densepose_rcnn_R_50_FPN_s1x.yaml)",
    )
    parser.add_argument(
        "--model",
        metavar="<model>",
        required=True,
        help="Path to the DensePose model weights (e.g., model_final_162be9.pkl)",
    )
    parser.add_argument(
        "--input",
        metavar="<input_image>",
        required=True,
        help="Path to the input image of a person.",
    )
    parser.add_argument(
        "--output",
        metavar="<output_image>",
        default="output_IUV.png",
        help="Path to save the resulting IUV map image.",
    )
    args = parser.parse_args()

    # 1. Setup the predictor
    predictor = setup_densepose_predictor(args.config, args.model)
    print(f"Using device: {predictor.model.device}")

    # 2. Load the input image
    print(f"Reading input image from {args.input}")
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        print(f"Error: Could not read image from {args.input}")
        return

    # 3. Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = predictor(image_bgr)["instances"]

    # 4. Extract and visualize the results
    # The extractor gets the I, U, V values from the model's output
    extractor = DensePoseResultExtractor()
    # The visualizer turns these IUV values into a colorful image
    visualizer = DensePoseResultsVisualizerWithTexture()
    
    # Extract the data
    data = extractor(outputs)

    # Create a black background to draw the IUV map on
    black_background = np.zeros_like(image_bgr, dtype=np.uint8)

    # Visualize the IUV map on the black background
    iuv_map_image = visualizer.visualize(black_background, data)

    # 5. Save the output
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cv2.imwrite(args.output, iuv_map_image)
    print(f"Successfully saved IUV map to {args.output}")

if __name__ == "__main__":
    main()