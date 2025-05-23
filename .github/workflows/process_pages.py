#!/usr/bin/env python3
import sys
import json
import re
import os


def clean_deepwiki_content(content):
    content = re.sub(r'DeepWiki.*\n', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def convert_img_tags_to_md(content):
    content = re.sub(
        r'<img[^>]*src="([^"]+)"[^>]*alt="([^"]*)"[^>]*>', r'![\2](\1)', content)
    content = re.sub(r'<img[^>]*src="([^"]+)"[^>]*>', r'![](\1)', content)
    return content


def process(input_path, output_dir):
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)
    pages = []
    result = data.get('result')
    if isinstance(result, dict) and 'content' in result:
        items = result['content']
        if isinstance(items, list):
            pages = [item.get('text', '')
                     for item in items if isinstance(item, dict)]
        else:
            pages = [items.get('text', '')]
    elif isinstance(result, str):
        pages = [result]

    os.makedirs(output_dir, exist_ok=True)
    created = []
    for i, text in enumerate(pages, 1):
        clean = clean_deepwiki_content(text)
        clean = convert_img_tags_to_md(clean)
        if len(clean) < 50:
            continue
        title = clean.splitlines()[0].lstrip('# ').strip() or f"Page_{i}"
        fname = re.sub(r'[^\w\-]', '_', title) + '.md'
        path = os.path.join(output_dir, fname)
        with open(path, 'w', encoding='utf-8') as fo:
            fo.write(f'# {title}\n\n{clean}')
        created.append(fname)
        print(f"✅ Created {fname}")
    if not created:
        print("❌ No pages procesadas.")
        sys.exit(1)
    print("📄 Ficheros creados:", created)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: process_pages.py <input.json> <output_dir>")
        sys.exit(1)
    process(sys.argv[1], sys.argv[2])
