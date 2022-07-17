REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .meta_controller import MetaMAC
REGISTRY["meta_mac"] = MetaMAC

from .madt_controller import MADTMAC
REGISTRY["madt_mac"] = MADTMAC