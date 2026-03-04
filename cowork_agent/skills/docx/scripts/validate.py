#!/usr/bin/env python3
"""Validate a .docx file structure and content."""
import sys
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

REQUIRED_FILES = [
    "[Content_Types].xml",
    "word/document.xml",
    "_rels/.rels",
]

EXPECTED_FILES = [
    "word/styles.xml",
    "word/_rels/document.xml.rels",
]

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

def validate(docx_path: str) -> dict:
    """Validate a .docx file and return issues found."""
    docx_path = Path(docx_path).resolve()
    issues = []
    warnings = []

    if not docx_path.exists():
        return {"valid": False, "issues": [f"File not found: {docx_path}"], "warnings": []}

    # Check it's a valid zip
    if not zipfile.is_zipfile(docx_path):
        return {"valid": False, "issues": ["Not a valid ZIP/DOCX file"], "warnings": []}

    with zipfile.ZipFile(docx_path, 'r') as zf:
        names = zf.namelist()

        # Check required files
        for req in REQUIRED_FILES:
            if req not in names:
                issues.append(f"Missing required file: {req}")

        # Check expected files
        for exp in EXPECTED_FILES:
            if exp not in names:
                warnings.append(f"Missing expected file: {exp}")

        # Validate XML in key files
        xml_files = [n for n in names if n.endswith('.xml') or n.endswith('.rels')]
        for xml_file in xml_files:
            try:
                content = zf.read(xml_file)
                ET.fromstring(content)
            except ET.ParseError as e:
                issues.append(f"Invalid XML in {xml_file}: {e}")

        # Check document.xml specifically
        if "word/document.xml" in names:
            try:
                content = zf.read("word/document.xml")
                root = ET.fromstring(content)
                body = root.find(f"{{{WORD_NS}}}body")
                if body is None:
                    issues.append("document.xml missing <w:body> element")
                else:
                    paragraphs = body.findall(f".//{{{WORD_NS}}}p")
                    if len(paragraphs) == 0:
                        warnings.append("Document has no paragraphs")
            except Exception as e:
                issues.append(f"Error parsing document.xml: {e}")

        # Check file size
        total_size = sum(info.file_size for info in zf.infolist())
        if total_size > 100 * 1024 * 1024:  # 100MB
            warnings.append(f"Large document: {total_size / 1024 / 1024:.1f}MB uncompressed")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "file_count": len(names),
        "file_size_bytes": docx_path.stat().st_size,
    }

def main():
    """CLI entry point for validating a .docx file structure."""
    if len(sys.argv) < 2:
        print("Usage: python validate.py <file.docx>")
        sys.exit(1)

    result = validate(sys.argv[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["valid"] else 1)

if __name__ == "__main__":
    main()
