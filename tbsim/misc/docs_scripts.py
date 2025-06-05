import csv

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
    
    
