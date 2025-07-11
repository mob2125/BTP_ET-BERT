from uer.targets.mlm_target import MlmTarget
from uer.targets.lm_target import LmTarget
from uer.targets.bert_target import BertTarget
from uer.targets.cls_target import ClsTarget
from uer.targets.bilm_target import BilmTarget
from uer.targets.albert_target import AlbertTarget
from uer.targets.seq2seq_target import Seq2seqTarget
from uer.targets.t5_target import T5Target
from uer.targets.prefixlm_target import PrefixlmTarget
from uer.targets.tiny_target import TinyBertTarget


str2target = {"bert": BertTarget, "mlm": MlmTarget, "lm": LmTarget,
              "bilm": BilmTarget, "albert": AlbertTarget, "seq2seq": Seq2seqTarget,
              "t5": T5Target, "cls": ClsTarget, "prefixlm": PrefixlmTarget, "tinybert": TinyBertTarget}

__all__ = ["BertTarget", "MlmTarget", "LmTarget", "BilmTarget", "AlbertTarget",
           "Seq2seqTarget", "T5Target", "ClsTarget", "PrefixlmTarget", "str2target"]
