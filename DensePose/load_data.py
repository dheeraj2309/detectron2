import torch
import pickle

# STEP 1: Import the custom class (so torch can find it)
from densepose.structures.chart_result import DensePoseChartResultWithConfidences

# STEP 2: Allow PyTorch to trust this class
torch.serialization.add_safe_globals([DensePoseChartResultWithConfidences])

# STEP 3: Load the file
checkpoint = torch.load("D:\\Project 2\\Densepose\\detectron2\\projects\\DensePose\\data.pkl", map_location='cpu', weights_only=False)
