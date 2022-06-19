REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .updet_agent import UPDeT
REGISTRY['updet'] = UPDeT

from .vanilla_transformer_agent import VanillaTransformer
REGISTRY['vanilla_transformer'] = VanillaTransformer

from .transformer_agg_agent import TransformerAggregationAgent
REGISTRY['transformer_aggregation'] = TransformerAggregationAgent

from .axial_transformer_agent import AxialTransformerAgent
REGISTRY['axial_transformer'] = AxialTransformerAgent