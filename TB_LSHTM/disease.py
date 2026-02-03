"""
Polymorphic disease model wrappers for tb_acf.

This module provides a small abstraction layer so callers can work with a
single interface while swapping the underlying intrahost disease model
(e.g. TBsim vs LSHTM).
"""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tb_acf as acf

from tbsim.disease import BaseDisease
from tbsim.disease import TB as TB  # re-export
from tbsim.disease import LSHTM as _LSHTM


@dataclass(frozen=True)
class LSHTM(_LSHTM):
    """
    LSHTM wrapper specialized for tb_acf.

    We provide the infection class from `tb_acf.tb_lshtm` so callers can just
    select LSHTM without providing `infection_cls` manually.
    """

    def __post_init__(self) -> None:
        infection_cls = acf.TB_LSHTM_Acute if self.acute else acf.TB_LSHTM
        object.__setattr__(self, "infection_cls", infection_cls)

    def configure_casefinding_test_sens(self, pcf: Any, act3: Any) -> None:
        # From Schwalb 2024 (Table S3) for NAAT (Xpert MTB/RIF, Ultra)
        pcf.pars.test_sens = {
            acf.TBSL.ASYMPTOMATIC: 0.0,
            acf.TBSL.SYMPTOMATIC: 0.909,
        }
        act3.pars.test_sens = {
            acf.TBSL.ASYMPTOMATIC: 0.775,
            acf.TBSL.SYMPTOMATIC: 0.909,
        }


def from_inhost(inhost: str, *, name: str = "tb") -> BaseDisease:
    """
    Factory for tb_acf to select the correct disease wrapper.
    """
    key = (inhost or "").strip().lower()
    if key == "tbsim":
        return TB(name=name)
    elif key in {"lshtm", "lshtm-acute"}:
        return LSHTM(name=name, acute=(key == "lshtm-acute"))
    else:
        raise ValueError(f"Unknown inhost model: {inhost!r} (expected 'TBsim', 'LSHTM', or 'LSHTM-Acute')")

