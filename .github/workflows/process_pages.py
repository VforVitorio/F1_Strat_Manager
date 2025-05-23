#!/usr/bin/env python3
import sys
import json
import re
import os


def clean_deepwiki_content(content):
    """Clean DeepWiki content while preserving image positioning and structure"""

    # Remove only specific DeepWiki UI elements, not content
    content = re.sub(r'.*?DeepWiki.*?\n', '', content, flags=re.IGNORECASE)
    content = re.sub(r'.*?Powered by Devin.*?\n', '',
                     content, flags=re.IGNORECASE)
    content = re.sub(r'.*?Share.*?\n', '', content, flags=re.IGNORECASE)
    content = re.sub(r'.*?Last indexed:.*?\n', '',
                     content, flags=re.IGNORECASE)
    content = re.sub(r'.*?Try DeepWiki.*?\n', '', content, flags=re.IGNORECASE)
    content = re.sub(r'.*?Auto-refresh not enabled yet.*?\n',
                     '', content, flags=re.IGNORECASE)

    # Remove navigation but preserve content structure
    content = re.sub(r'Menu\n', '', content)
    content = re.sub(r'### On this page.*?- Getting Started\n',
                     '', content, flags=re.DOTALL)

    # Remove source file references but keep the actual content
    content = re.sub(r'Relevant source files.*?\n\n',
                     '', content, flags=re.DOTALL)
    content = re.sub(r'Sources:.*?\n', '', content, flags=re.MULTILINE)

    # Clean up title duplicates
    content = re.sub(r'^# /VforVitorio/F1_Strat_Manager.*?\n',
                     '', content, flags=re.MULTILINE)
    content = re.sub(
        r'VforVitorio/F1_Strat_Manager \| DeepWiki.*?\n', '', content)

    # Be more conservative with newline reduction - preserve structure around images
    # Only reduce excessive newlines (4 or more) but preserve intentional spacing
    content = re.sub(r'\n{5,}', '\n\n\n', content)

    # Preserve double spacing around important elements (but not triple+)
    # This helps maintain image positioning
    content = re.sub(r'(\n\n)\n+(\n)', r'\1\2', content)

    return content.strip()


def preserve_image_positioning(content):
    """Convert image tags to markdown while preserving exact positioning"""

    # First, let's identify and preserve the context around images
    img_pattern = r'(<img[^>]*src="([^"]+)"[^>]*(?:alt="([^"]*)"[^>]*)?>[^<]*)'

    def img_replacer(match):
        full_match = match.group(0)
        src = match.group(2)
        alt = match.group(3) if match.group(3) else "Diagram"

        # Convert to markdown format
        return f'![{alt}]({src})'

    # Replace image tags with markdown, preserving position
    content = re.sub(img_pattern, img_replacer, content)

    # Handle simpler img tags without groups
    simple_img_pattern = r'<img[^>]*src="([^"]+)"[^>]*(?:alt="([^"]*)"[^>]*)?>'

    def simple_img_replacer(match):
        src = match.group(1)
        alt = match.group(2) if len(
            match.groups()) > 1 and match.group(2) else "System Diagram"
        return f'![{alt}]({src})'

    content = re.sub(simple_img_pattern, simple_img_replacer, content)

    # Preserve spacing around images - don't compress whitespace immediately after images
    content = re.sub(
        r'(!\[[^\]]*\]\([^)]+\))\s*\n\s*\n\s*\n+', r'\1\n\n', content)

    # Ensure images have proper spacing from surrounding content
    content = re.sub(r'(\w)\s*(!\[[^\]]*\]\([^)]+\))', r'\1\n\n\2', content)
    content = re.sub(r'(!\[[^\]]*\]\([^)]+\))\s*(\w)', r'\1\n\n\2', content)

    return content


def add_contextual_diagrams(content):
    """Add contextual Mermaid diagrams where appropriate"""

    # Only add diagrams where there are clear gaps or references to missing visuals
    diagram_patterns = {
        # System architecture mentions
        r'(system.*?(?:architecture|component|structure).*?)\n\n\n+':
            r'\1\n\n```mermaid\ngraph TB\n    subgraph "Data Sources"\n        A[FastF1 API]\n        B[OpenF1 Radio]\n        C[Video Stream]\n    end\n    \n    subgraph "Processing"\n        D[Data Processor]\n        E[Feature Engineering]\n    end\n    \n    subgraph "Analysis"\n        F[ML Models]\n        G[Computer Vision]\n        H[NLP Pipeline]\n    end\n    \n    subgraph "Strategy"\n        I[Expert System]\n        J[Rule Engine]\n    end\n    \n    A --> D\n    B --> D\n    C --> D\n    D --> E\n    E --> F\n    E --> G\n    E --> H\n    F --> I\n    G --> I\n    H --> I\n    I --> J\n```\n\n',

        # Data flow mentions
        r'(data.*?flow.*?through.*?)\n\n\n+':
            r'\1\n\n```mermaid\nflowchart LR\n    A[Input Data] --> B[Processing]\n    B --> C[Analysis]\n    C --> D[Strategy Output]\n```\n\n',

        # Generic empty spaces after diagram mentions
        r'(.*?(?:diagram|figure|chart|visual).*?)\n\n\n\n+':
            r'\1\n\n*(Diagram placeholder - see [DeepWiki](https://deepwiki.com/VforVitorio/F1_Strat_Manager) for original visual)*\n\n',
    }

    for pattern, replacement in diagram_patterns.items():
        content = re.sub(pattern, replacement, content,
                         flags=re.IGNORECASE | re.DOTALL)

    return content


def extract_section_title(content):
    """Extract meaningful title from content"""
    lines = content.split('\n')

    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if line.startswith('# ') and len(line) > 3:
            title = line[2:].strip()
            # Clean up the title
            title = re.sub(r'^/.*?/', '', title)  # Remove leading path
            title = re.sub(r'[^\w\s-]', '', title)  # Remove special chars
            if title and title != 'VforVitorio' and len(title) > 2:
                return title

    # Fallback: look for meaningful content
    for line in lines[:20]:
        line = line.strip()
        if line.startswith('##') and len(line) > 4:
            title = line.lstrip('#').strip()
            if len(title) > 3:
                return title

    return "F1 Strategy Manager Documentation"


def generate_safe_filename(title):
    """Generate safe filename from title"""
    filename = re.sub(r'[^\w\s-]', '', title.lower())
    filename = re.sub(r'[-\s]+', '-', filename)
    filename = filename.strip('-')

    if not filename or len(filename) < 3:
        filename = "documentation"

    return filename + ".md"


def process(input_path, output_dir):
    """Process DeepWiki content preserving structure and image positioning"""

    try:
        with open(input_path, encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        # Try as plain text
        with open(input_path, encoding='utf-8') as f:
            content = f.read()

        # Process as single page
        pages = [content]
        print("📄 Processing as plain text...")
    else:
        # Extract pages from JSON structure
        pages = []
        result = data.get('result')

        if isinstance(result, dict) and 'content' in result:
            items = result['content']
            if isinstance(items, list):
                pages = [item.get('text', '') for item in items if isinstance(
                    item, dict) and item.get('text')]
            elif isinstance(items, dict) and 'text' in items:
                pages = [items['text']]
        elif isinstance(result, str):
            pages = [result]

        print(f"📄 Found {len(pages)} content sections")

    os.makedirs(output_dir, exist_ok=True)
    created_files = []    # Process each page/section
    for i, raw_content in enumerate(pages, 1):
        if len(raw_content.strip()) < 50:
            print(f"⏭️ Skipping short content section {i}")
            continue

        print(f"🔄 Processing section {i} ({len(raw_content)} chars)...")

        # Debug: Check for images in original content
        img_count_original = len(re.findall(r'<img[^>]*>', raw_content))
        print(f"   📷 Original images found: {img_count_original}")

        # Clean content while preserving structure
        content = clean_deepwiki_content(raw_content)

        # Debug: Check for images after cleaning
        img_count_after_clean = len(re.findall(r'<img[^>]*>', content))
        print(f"   📷 Images after cleaning: {img_count_after_clean}")

        # Convert images while preserving positioning
        content = preserve_image_positioning(content)

        # Debug: Check for markdown images after conversion
        md_img_count = len(re.findall(r'!\[[^\]]*\]\([^)]+\)', content))
        print(f"   📷 Markdown images after conversion: {md_img_count}")

        # Add contextual diagrams where appropriate
        content = add_contextual_diagrams(content)

        # Debug: Final image count
        final_img_count = len(re.findall(r'!\[[^\]]*\]\([^)]+\)', content))
        mermaid_count = len(re.findall(r'```mermaid', content))
        print(
            f"   📷 Final images: {final_img_count}, Mermaid diagrams: {mermaid_count}")

        # Extract title and generate filename
        title = extract_section_title(content)
        filename = generate_safe_filename(title)

        # Ensure content starts with proper title
        if not content.strip().startswith('# '):
            content = f'# {title}\n\n{content.strip()}'

        # Write file
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        created_files.append((filename, title))
        print(f"✅ Created {filename} - {title}")

    if not created_files:
        print("❌ No pages processed.")
        sys.exit(1)

    # Create comprehensive documentation
    print("📚 Creating comprehensive documentation...")

    main_content = "# F1 Strategy Manager - Complete Documentation\n\n"
    main_content += "This document contains the complete documentation for the F1 Strategy Manager project.\n\n"
    main_content += "> **Note**: Some diagrams may be represented as Mermaid diagrams or placeholders. For the original visual representations, please visit the [DeepWiki documentation](https://deepwiki.com/VforVitorio/F1_Strat_Manager).\n\n"

    # Table of contents
    main_content += "## Table of Contents\n\n"
    for filename, title in created_files:
        section_link = title.lower().replace(' ', '-').replace('.',
                                                               '').replace('(', '').replace(')', '')
        main_content += f"- [{title}](#{section_link})\n"
    main_content += "\n---\n\n"

    # Add all content
    for filename, title in created_files:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        main_content += content + "\n\n---\n\n"

    # Write comprehensive file
    comprehensive_path = os.path.join(
        output_dir, 'f1-strat-manager-complete.md')
    with open(comprehensive_path, 'w', encoding='utf-8') as f:
        f.write(main_content)

    print(f"✅ Created comprehensive documentation: f1-strat-manager-complete.md")
    print(f"📄 Total files created: {len(created_files) + 1}")
    print(
        f"📁 Files: {[f[0] for f in created_files] + ['f1-strat-manager-complete.md']}")

    return created_files


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: process_pages.py <input.json> <output_dir>")
        sys.exit(1)
    process(sys.argv[1], sys.argv[2])
