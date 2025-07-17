# pdf_to_json.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Literal

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat


# 2. Re-usable converter (global so we don’t recreate it every call)
_PIPELINE = PdfPipelineOptions(
    do_table_structure=True,
    table_structure_options={"mode": TableFormerMode.FAST},
    enable_remote_services=False,
)
_CONVERTER = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_PIPELINE)}
)

# 3. Public helper

def pdf_to_json(
    pdf_path: str | Path,
    *,
    write_file: bool = False,
    out_dir: str | Path | None = None,
    indent: int | None = None,
) -> str:
    """
    Convert a single PDF → JSON string (Docling enriched format).
    Optionally write `<stem>.json` next to the PDF or into `out_dir`.

    Returns
    -------
    str
        The raw JSON string (utf-8).
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    # Convert
    result = _CONVERTER.convert(str(pdf_path))
    if not result.document:
        raise RuntimeError(f"Docling failed to convert {pdf_path}")

    # Export
    doc_dict = result.document.export_to_dict()
    json_str = json.dumps(doc_dict, ensure_ascii=False, indent=indent)

    # Optional write
    if write_file:
        out_path = (
            Path(out_dir).expanduser().resolve() / f"{pdf_path.stem}.json"
            if out_dir
            else pdf_path.with_suffix(".json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")

    return json_str

# 4. CLI convenience:  python pdf_to_json.py file.pdf

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pdf_to_json.py <path/to/file.pdf>")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    js = pdf_to_json(pdf, write_file=True)  # writes file.pdf.json next to PDF
    print(f"Converted {pdf.name} → {len(js)} chars of JSON")