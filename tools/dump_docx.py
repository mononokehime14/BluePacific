"""把 PTQ技术.docx 的正文（含表格）按文档顺序导出成 text，便于 diff / 检索。"""

import sys

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(parent):
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


doc = Document(sys.argv[1] if len(sys.argv) > 1 else "PTQ技术.docx")
out = []
for block in iter_block_items(doc):
    if isinstance(block, Paragraph):
        t = block.text.rstrip()
        if t:
            style = block.style.name
            prefix = f"[{style}] " if style != "Normal" else ""
            out.append(prefix + t)
    else:
        out.append("<<TABLE>>")
        for row in block.rows:
            cells = [c.text.strip().replace("\n", " / ") for c in row.cells]
            out.append(" | ".join(cells))
        out.append("<<END TABLE>>")

with open(".tmp/ptq_doc.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("lines:", len(out))
