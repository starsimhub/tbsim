from enum import Enum
from typing import List, Dict
import csv

__all__ = ["TBTerms", "TermsExport"]
class TBTerms(Enum):
    """
    Enumeration of tuberculosis-related terms and their full descriptions.

    This enum provides:
    - Safe, structured access to common TB-related abbreviations.
    - Reverse lookup using original terms such as 'RR-TB' or '1HP'.
    - Integration with tools for tooltips, autocomplete, and documentation.
    - Friendly string representations for use in UIs or logs.

    Example:
        >>> TBTerms.RR.name
        'RR'
        >>> TBTerms.RR.value
        'relative risk'
        >>> TBTerms._1HP.orig()
        '1HP'
        >>> TBTerms.from_key("RR_TB").value
        'rifampicin-resistant TB'
        >>> TBTerms.as_dict()["TPT"]
        'tuberculosis preventive treatment'
    """

    _1HP = "1 month of daily rifapentine plus isoniazid"
    _3HP = "3 months of weekly rifapentine plus isoniazid"
    _3HR = "3 months of daily rifampicin plus isoniazid"
    _4R = "4 months of daily rifampicin monotherapy"
    _6H = "6 months of daily isoniazid monotherapy"
    _6Lfx = "6 months of daily levofloxacin monotherapy"
    _9H = "9 months of daily isoniazid monotherapy"
    mtb = "Mycobacterium tuberculosis"
    ACF = "active case finding"
    ART = "antiretroviral therapy"
    ARV = "antiretroviral drugs"
    BCG = "bacille Calmette-Guérin"
    CAD = "computer-aided detection"
    CRP = "C-reactive protein"
    CXR = "chest radiography"
    DSD = "differentiated HIV service delivery"
    ELISA = "enzyme-linked immunosorbent assay"
    FDC = "fixed-dose combination"
    GDG = "Guideline Development Group"
    HMIS = "health management information system"
    IFN_γ = "interferon-γ"
    IGRA = "interferon-γ release assay"
    IPT = "isoniazid preventive treatment"
    LF = "latent fast TB infection"
    LFT = "liver function test"
    Lfx = "levofloxacin"
    LS = "latent slow TB infection"
    M_E = "monitoring and evaluation"
    MDR_TB = "multidrug-resistant tuberculosis"
    mWRD = "molecular WHO-recommended rapid diagnostic test"
    NGO = "nongovernmental organization"
    NNRTI = "non-nucleoside reverse transcriptase inhibitor"
    NRTI = "nucleotide reverse transcriptase inhibitor"
    PI = "protease inhibitor"
    PMTPT = "programmatic management of tuberculosis preventive treatment"
    PPD = "purified protein derivative"
    RCT = "randomized controlled trial"
    RR = "relative risk"
    RR_TB = "rifampicin-resistant TB"
    SMNEG = "Smear-negative TB"
    SMPOS = "Smear-positive TB"
    SOP = "standard operating procedure"
    TB = "tuberculosis"
    TBST = "Mycobacterium tuberculosis antigen-based skin test"
    TDF = "tenofovir-disoproxil fumarate"
    TNF = "tumour necrosis factor"
    TPT = "tuberculosis preventive treatment"
    XPTB = "extrapulmonary TB"

    def orig(self) -> str:
        """
        Return the original representation of the abbreviation.

        Returns:
            str: Original key, e.g., '1HP', 'RR-TB'.

        Example:
            >>> TBTerms._3HP.orig()
            '3HP'
        """
        name = self.name
        if name.startswith("_"):
            name = name[1:]
        return name.replace("_", "-") if "-" in name or "_" in self.name else name

    def help(self) -> str:
        """
        Return a human-friendly string representation.

        Returns:
            str: Label in the form 'KEY: description'.

        Example:
            >>> TBTerms.RR.help()
            'RR: relative risk'
        """
        return f"{self.orig()}: {self.value}"

    def __str__(self) -> str:
        """String representation for display purposes."""
        return self.help()

    def __repr__(self) -> str:
        """Debug-friendly representation."""
        return f"<{self.__class__.__name__}.{self.name}: '{self.value}'>"

    @classmethod
    def get(cls, key: str) -> "TBTerms":
        """
        Retrieve an enum item from the original key format.

        Args:
            key (str): Key such as 'RR-TB', '1HP', etc.

        Returns:
            TBTerms: Corresponding enum entry.

        Raises:
            TypeError: If the input is not a string.
            KeyError: If no matching key is found.

        Example:
            >>> TBTerms.from_key("TDF")
            <TBTerms.TDF: 'tenofovir-disoproxil fumarate'>
        """
        if not isinstance(key, str):
            raise TypeError(f"Expected string, got {type(key).__name__}")
        key_fmt = key.replace("-", "_")
        if key_fmt and key_fmt[0].isdigit():
            key_fmt = "_" + key_fmt
        try:
            return cls[key_fmt]
        except KeyError:
            valid_keys = [term.orig() for term in cls]
            raise KeyError(f"'{key}' is not a valid TB term. Valid options include: {valid_keys}")

    @classmethod
    def keys(cls) -> List[str]:
        """
        List all internal enum-safe keys.

        Returns:
            List[str]: Enum member names.

        Example:
            >>> TBTerms.keys()[:3]
            ['_1HP', '_3HP', '_3HR']
        """
        return [member.name for member in cls]

    @classmethod
    def values(cls) -> List[str]:
        """
        List all term descriptions.

        Returns:
            List[str]: Values of all terms.
        """
        return [member.value for member in cls]

    @classmethod
    def as_dict(cls) -> Dict[str, str]:
        """
        Return a dictionary mapping original keys to descriptions.

        Returns:
            Dict[str, str]: Original key → description.
        """
        return {member.orig(): member.value for member in cls}

class TermsExport():
    @staticmethod
    def export_tbterms_to_csv(filepath: str) -> None:
        """
        Write the TBTerms glossary to a CSV file.

        Args:
            filepath (str): Path to write the file.
        """
        from tbsim.misc.tbterms import TBTerms  # Lazy import to avoid circular dependency
        with open(filepath, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Term", "Description"])
            for term in TBTerms:
                writer.writerow([term.orig(), term.value])

    @staticmethod
    def export_tbterms_to_markdown() -> str:
        """
        Generate a markdown-formatted glossary of all TBTerms.

        Returns:
            str: A Markdown string with a glossary table.
        """
        from tbsim.misc.tbterms import TBTerms  # Lazy import to avoid circular dependency
        header = "| Term | Description |\n|------|-------------|\n"
        rows = [f"| {term.orig()} | {term.value} |" for term in TBTerms]
        return header + "\n".join(rows)

        # Save to file or print
        print(export_tbterms_to_markdown())