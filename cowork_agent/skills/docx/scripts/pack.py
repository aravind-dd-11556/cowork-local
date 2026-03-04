#!/usr/bin/env python3
"""Repack a directory back into a .docx file."""
import sys
import os
import zipfile
from pathlib import Path

def pack(source_dir: str, output_path: str = None) -> str:
    """Pack a directory back into a .docx file.

    Args:
        source_dir: Path to the unpacked docx directory
        output_path: Output .docx path (default: source_dir + .docx)

    Returns:
        Path to the created .docx file
    """
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Not a directory: {source}")

    # Verify it looks like an unpacked docx
    content_types = source / "[Content_Types].xml"
    if not content_types.exists():
        raise ValueError(f"Missing [Content_Types].xml - not a valid unpacked docx: {source}")

    if output_path is None:
        output_path = str(source) + ".docx"

    output = Path(output_path).resolve()

    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(source)
                zf.write(file_path, arcname)

    print(f"Packed {source.name} to {output.name} ({output.stat().st_size} bytes)")
    return str(output)

def main():
    if len(sys.argv) < 2:
        print("Usage: python pack.py <unpacked_dir> [output.docx]")
        sys.exit(1)

    source_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = pack(source_dir, output_path)
        print(f"Success: {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
