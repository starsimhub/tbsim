"""
Disease model selection for tbsim.

Single interface to choose the intrahost TB model (main TB vs LSHTM) and,
for LSHTM, to set case-finding test sensitivities by TBSL state.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import tbsim as mtb

__all__ = ['BaseDisease', 'TB', 'LSHTM', 'from_inhost']

class BaseDisease(Protocol):
    """Protocol for a disease wrapper with an infection class and optional test-sens config."""

    name: str
    infection_cls: type

    def configure_casefinding_test_sens(self, pcf: Any) -> None:
        """Set test sensitivity by disease state on pcf.pars.test_sens (optional)."""
        ...


@dataclass(frozen=True)
class TB:
    """Wrapper that uses the main tbsim TB model (tbsim.tb.TB) as infection_cls."""

    name: str = "tb"
    infection_cls: type = mtb.TB

    def configure_casefinding_test_sens(self, pcf: Any) -> None:
        pass  # Main TB uses its own state enum; override in subclasses if needed


@dataclass(frozen=True)
class LSHTM(TB):
    """Wrapper that uses TB_LSHTM or TB_LSHTM_Acute as infection_cls (acute=True for Acute)."""

    acute: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "infection_cls",
            mtb.TB_LSHTM_Acute if self.acute else mtb.TB_LSHTM,
        )

    def configure_casefinding_test_sens(self, pcf: Any) -> None:
        # Schwalb 2024 Table S3 — NAAT (Xpert MTB/RIF, Ultra)
        pcf.pars.test_sens = {
            mtb.TBSL.ASYMPTOMATIC: 0.0,
            mtb.TBSL.SYMPTOMATIC: 0.909,
        }


def from_inhost(inhost: str, *, name: str = "tb") -> BaseDisease:
    """Return the disease wrapper for the given in-host model name (tbsim, lshtm, lshtm-acute)."""
    key = (inhost or "").strip().lower()
    if key == "tbsim":
        return TB(name=name)
    if key in ("lshtm", "lshtm-acute"):
        return LSHTM(name=name, acute=(key == "lshtm-acute"))
    raise ValueError(
        f"Unknown inhost model: {inhost!r} (expected 'tbsim', 'lshtm', or 'lshtm-acute')"
    )
