"""Models package for baselines and advanced models."""

from .baseline import build_logistic_pipeline, train_and_evaluate
from .embeddings import ArkoudiStyleLogit
from .video_backbone_lstm import FrameBackboneLSTM
from .video_torch import VideoCNNLSTM, ArkoudiHead
from .mm_vae import DeterministicMMVAE, VariationalMMVAE
