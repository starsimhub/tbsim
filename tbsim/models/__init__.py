"""
TB disease models for Starsim.

Import model classes from this package so that module filenames are not needed:

  from tbsim.models import TB, TBS
  from tbsim.models import TB_LSHTM, TB_LSHTM_Acute, TBSL
"""

from .tb import TB, TBS
from .tb_lshtm import TB_LSHTM, TB_LSHTM_Acute, TBSL

__all__ = [
    'TB',
    'TBS',
    'TB_LSHTM',
    'TB_LSHTM_Acute',
    'TBSL',
]
