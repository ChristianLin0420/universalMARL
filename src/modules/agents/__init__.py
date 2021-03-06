REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .updet_agent import UPDeT
REGISTRY['updet'] = UPDeT

from .dummy_updet_agent import DummyUPDeT
REGISTRY['dummy_updet'] = DummyUPDeT

from .vanilla_transformer_agent import VanillaTransformer
REGISTRY['vanilla_transformer'] = VanillaTransformer

from .dummy_transformer_agent import DummyTransformer
REGISTRY['dummy_transformer'] = DummyTransformer

from .transformer_agg_agent import TransformerAggregationAgent
REGISTRY['transformer_aggregation'] = TransformerAggregationAgent

from .axial_transformer_agent import AxialTransformerAgent
REGISTRY['axial_transformer'] = AxialTransformerAgent

TRANSFORMERbasedAgent = ['updet', 'dummy_updet', 'transformer_aggregation', 'axial_transformer', 'vanilla_transformer', 'dummy_transformer']