REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .updet_agent import UPDeT
REGISTRY['updet'] = UPDeT

from .vanilla_transformer_agent import VanillaTransformer
REGISTRY['vanilla_transformer'] = VanillaTransformer

from .dummy_transformer_agent import DummyTransformer
REGISTRY['dummy_transformer'] = DummyTransformer

from .transformer_agg_agent import TransformerAggregationAgent
REGISTRY['transformer_aggregation'] = TransformerAggregationAgent

from .axial_transformer_agent import AxialTransformerAgent
REGISTRY['axial_transformer'] = AxialTransformerAgent