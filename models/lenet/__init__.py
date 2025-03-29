#!/usr/bin/env python3
"""
LeNet model for breast tumor classification.
"""

from .src.model import LeNet
from .src.dataset import BreastTumorDataset, get_dataloaders