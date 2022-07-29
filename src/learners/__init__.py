from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .madt_learner import MADTLearner
from .rode_learner import RODELearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["madt_learner"] = MADTLearner
REGISTRY["rode_learner"] = RODELearner