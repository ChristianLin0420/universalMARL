REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .meta_runner import MetaRunner
REGISTRY["meta"] = MetaRunner

from .madt_runner import MADTRunner
REGISTRY["madt"] = MADTRunner
