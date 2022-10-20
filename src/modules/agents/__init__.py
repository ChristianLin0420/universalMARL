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

from .transfermer_agent import Transfermer
REGISTRY['transfermer'] = Transfermer

from .transfermer_plus_agent import TransfermerPlus
REGISTRY['transfermer++'] = TransfermerPlus

from .transformer_agg_agent import TransformerAggregationAgent
REGISTRY['transformer_aggregation'] = TransformerAggregationAgent

from .axial_transformer_agent import AxialTransformerAgent
REGISTRY['axial_transformer'] = AxialTransformerAgent

from .perceiverIO_agent import PerceiverIOAgent
REGISTRY['perceiver_io'] = PerceiverIOAgent

from .perceiverIOplus_agent import PerceiverIOplusAgent
REGISTRY['perceiver++'] = PerceiverIOplusAgent

from .double_perceiver_agent import DoublePerceiverAgent
REGISTRY['double_perceiver'] = DoublePerceiverAgent

from .trackformer_agent import TrackformerAgent
REGISTRY['trackformer'] = TrackformerAgent

from .gpt_agent import GPTAgent
REGISTRY['gpt'] = GPTAgent

from .fuseformer_agent import FouseformerAgent
REGISTRY['fuseformer'] = FouseformerAgent

TRANSFORMERbasedAgent = [   'updet', 
                            'dummy_updet', 
                            'transformer_aggregation', 
                            'axial_transformer', 
                            'vanilla_transformer', 
                            'dummy_transformer', 
                            'transfermer', 
                            'transfermer++',
                            'perceiver_io', 
                            'perceiver++',
                            'double_perceiver',
                            'trackformer', 
                            'gpt', 
                            'fuseformer'   ]