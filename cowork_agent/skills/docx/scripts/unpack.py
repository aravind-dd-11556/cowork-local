#!/usr/bin/env python3
"""Unpack a .docx file into a directory for XML editing."""
import sys
import os
import zipfile
from pathlib import Path

def unpack(docx_path: str, output_dir: str = None) -> str:
    """Unpack a .docx file to a directory.

    Args:
        docx_path: Path to the .docx file
        output_dir: Output directory (default: same name without extension)

    Returns:
        Path to the unpacked directory
    """
    docx_path = Path(docx_path).resolve()
    if not docx_path.exists():
        raise FileNotFoundError(f"File not found: {docx_path}")
    if not docx_path.suffix.lower() == '.docx':
        raise ValueError(f"Not a .docx file: {docx_path}")

    if output_dir is None:
        output_dir = str(docx_path.with_suffix(''))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(docx_path, 'r') as zf:
        zf.extractall(output_path)

    print(f"Unpacked {docx_path.name} to {output_path}")
    print(f"Key files:")
    for key_file in ["word/document.xml", "word/styles.xml", "word/numbering.xml",
                      "[Content_Types].xml", "word/_rels/document.xml.rels"]:
        full = output_path / key_file
        if full.exists():
            print(f"  {key_file} ({full.stat().st_size} bytes)")

    return str(output_path)

def main():
    """CLI entry point for unpacking a .docx file into a directory."""
    if len(sys.argv) < 2:
        print("Usage: python unpack.py <file.docx> [output_dir]")
        sys.exit(1)

    docx_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = unpack(docx_path, output_dir)
        print(f"\nSuccess: {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
