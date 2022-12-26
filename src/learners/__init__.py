from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .transformer_learner import TransLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["trans_learner"] = TransLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner