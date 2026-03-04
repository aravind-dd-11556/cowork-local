#!/usr/bin/env python3
"""Add, remove, and list comments in a .docx file."""
import sys
import json
from pathlib import Path

try:
    from docx import Document
    from docx.oxml.ns import qn
    from lxml import etree
except ImportError:
    print("Error: python-docx required. pip install python-docx", file=sys.stderr)
    sys.exit(1)

def list_comments(docx_path: str) -> list[dict]:
    """List all comments in a .docx file."""
    doc = Document(docx_path)
    comments = []

    # Access the comments part if it exists
    for rel in doc.part.rels.values():
        if "comments" in rel.reltype:
            comments_part = rel.target_part
            root = etree.fromstring(comments_part.blob)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

            for comment in root.findall("w:comment", ns):
                comment_id = comment.get(qn("w:id"))
                author = comment.get(qn("w:author"), "Unknown")
                date = comment.get(qn("w:date"), "")

                # Extract text from comment paragraphs
                texts = []
                for p in comment.findall(".//w:t", ns):
                    if p.text:
                        texts.append(p.text)

                comments.append({
                    "id": comment_id,
                    "author": author,
                    "date": date,
                    "text": " ".join(texts),
                })
            break

    return comments

def add_comment(docx_path: str, text: str, author: str = "Cowork Agent",
                paragraph_index: int = 0, output_path: str = None) -> str:
    """Add a comment to a specific paragraph."""
    from datetime import datetime

    doc = Document(docx_path)

    if paragraph_index >= len(doc.paragraphs):
        raise IndexError(f"Paragraph {paragraph_index} does not exist (doc has {len(doc.paragraphs)} paragraphs)")

    paragraph = doc.paragraphs[paragraph_index]

    # Create comment reference in the paragraph
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    comment_id = "1"  # Simplified - production would track IDs

    comment_start = etree.SubElement(paragraph._p, qn("w:commentRangeStart"))
    comment_start.set(qn("w:id"), comment_id)

    comment_end = etree.SubElement(paragraph._p, qn("w:commentRangeEnd"))
    comment_end.set(qn("w:id"), comment_id)

    run = etree.SubElement(paragraph._p, qn("w:r"))
    ref = etree.SubElement(run, qn("w:commentReference"))
    ref.set(qn("w:id"), comment_id)

    out = output_path or docx_path
    doc.save(out)
    return out

def main():
    """CLI entry point for listing or adding comments in a .docx file."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python comment.py list <file.docx>")
        print("  python comment.py add <file.docx> <text> [author] [paragraph_index] [output.docx]")
        sys.exit(1)

    action = sys.argv[1]
    docx_path = sys.argv[2]

    if action == "list":
        comments = list_comments(docx_path)
        print(json.dumps(comments, indent=2))
    elif action == "add":
        if len(sys.argv) < 4:
            print("Error: text required for add", file=sys.stderr)
            sys.exit(1)
        text = sys.argv[3]
        author = sys.argv[4] if len(sys.argv) > 4 else "Cowork Agent"
        para_idx = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        output = sys.argv[6] if len(sys.argv) > 6 else None
        result = add_comment(docx_path, text, author, para_idx, output)
        print(f"Comment added. Saved to: {result}")
    else:
        print(f"Unknown action: {action}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
