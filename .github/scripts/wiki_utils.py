#!/usr/bin/env python3
"""
Utilidades para el procesamiento de contenido de DeepWiki
"""

import re
import os
import urllib.parse
import subprocess
from pathlib import Path


def clean_deepwiki_content(content):
    """Clean and format DeepWiki content for GitHub Wiki"""
    # Remove navigation menu - improved patterns for better coverage
    nav_patterns = [
        # Pattern 1: Complete navigation menu with all sections
        r'(?s)^.*?(?:Overview\n.*?System Architecture.*?Installation and Setup.*?Streamlit Dashboard.*?Strategy Recommendations View.*?Gap Analysis View.*?Radio Analysis View.*?Time Predictions View.*?Strategy Chat Interface.*?Machine Learning Models.*?Lap Time Prediction.*?Tire Degradation Modeling.*?Vision-based Gap Calculation.*?NLP Pipeline.*?Radio Transcription.*?Sentiment and Intent Analysis.*?Named Entity Recognition.*?Expert System.*?Degradation Rules.*?Gap Analysis Rules.*?Radio Message Rules.*?Integrated Rule Engine.*?Developer Guide.*?API Reference.*?Integration Guide).*?(?=\n#{1,3}\s+[^#]|\n[A-Z][^\n]*\n|\Z)',

        # Pattern 2: Shorter navigation menus
        r'(?s)^.*?(?:Overview|System Architecture|Streamlit Dashboard|Machine Learning Models|NLP Pipeline|Expert System|Developer Guide).*?(?:Overview|System Architecture|Streamlit Dashboard|Machine Learning Models|NLP Pipeline|Expert System|Developer Guide).*?(?=\n#{1,3}\s+[^#]|\n[A-Z][^\n]*\n|\Z)',

        # Pattern 3: Bullet-point navigation
        r'(?m)^(?:\s*[-•]\s*(?:Overview|System Architecture|Streamlit Dashboard|Machine Learning Models|NLP Pipeline|Expert System|Developer Guide|Getting Started).*?\n)+',

        # Pattern 4: Menu headers and "On this page" sections
        r'(?s)(?:Menu\n|### On this page.*?)(?=\n#{1,3}\s+[^#]|\n[A-Z][^\n]*\n|\Z)',

        # Pattern 5: DeepWiki revision info
        r'(?m)^.*?github-actions\[bot\].*?revision.*?\n',
    ]

    for pattern in nav_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    # Remove section dividers and repeated slashes
    content = re.sub(r'^#? ?/?(?:\n/+)*', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*/+\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^/?\n+', '', content, flags=re.MULTILINE)

    # Remove DeepWiki UI elements
    ui_elements = [
        r'.*?DeepWiki.*?\n',
        r'.*?Powered by Devin.*?\n',
        r'.*?Share.*?\n',
        r'.*?Last indexed:.*?\n',
        r'.*?Try DeepWiki.*?\n',
        r'.*?Auto-refresh not enabled yet.*?\n',
        r'.*?Which repo would you like to understand.*?\n'
    ]

    for pattern in ui_elements:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    # Remove source file references (they clutter the wiki)
    content = re.sub(r'Relevant source files.*?\n\n',
                     '', content, flags=re.DOTALL)
    content = re.sub(r'Sources:.*?\n', '', content, flags=re.MULTILINE)

    # Clean up multiple consecutive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Remove empty sections
    content = re.sub(r'\n## \n', '', content)
    content = re.sub(r'\n### \n', '', content)

    # Fix malformed headers
    content = re.sub(r'^([#]+)\s*$', '', content, flags=re.MULTILINE)

    # Remove VforVitorio/F1_Strat_Manager title duplicates
    content = re.sub(r'^# /VforVitorio/F1_Strat_Manager.*?\n',
                     '', content, flags=re.MULTILINE)
    content = re.sub(
        r'VforVitorio/F1_Strat_Manager \| DeepWiki.*?\n', '', content)

    # Clean up beginning of content
    content = content.strip()

    return content


def get_page_title_from_content(content):
    """Extract a clean title from content"""
    lines = content.split('\n')
    for line in lines:
        if line.strip().startswith('# ') and len(line.strip()) > 2:
            title = line.strip()[2:].strip()
            # Clean the title
            title = re.sub(r'^/.*?/', '', title)  # Remove leading path
            if title:
                return title
    return "Documentation"


def safe_filename(title, fallback_idx=None):
    """Genera un nombre de archivo seguro a partir del título"""
    if not title or len(title.strip()) == 0:
        if fallback_idx is not None:
            return f"unknown-section-{fallback_idx}"
        return "unknown-section"

    filename = title.lower().strip()
    # Elimina caracteres especiales y reemplaza por guiones
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    filename = filename.strip('-')  # Quita guiones al inicio/fin
    if not filename:
        if fallback_idx is not None:
            return f"unknown-section-{fallback_idx}"
        return "unknown-section"
    return filename


def categorize_section(title, content="", fallback_idx=None):
    """Categoriza secciones según la jerarquía especificada"""
    title_lower = title.lower() if title else ''
    content_lower = content.lower() if content else ''

    # Main sections
    if any(keyword in title_lower for keyword in ['overview', 'introducción', 'introduction', 'main', 'general']):
        return ('main', 'Overview', '01-overview.md')
    elif any(keyword in title_lower for keyword in ['streamlit', 'dashboard', 'interfaz', 'ui', 'interface']):
        return ('main', 'Streamlit Dashboard', '02-streamlit-dashboard.md')
    elif any(keyword in title_lower for keyword in ['machine learning', 'ml', 'modelo', 'model', 'prediction']):
        return ('main', 'Machine Learning Models', '03-machine-learning-models.md')
    elif any(keyword in title_lower for keyword in ['nlp', 'natural language', 'radio', 'processing', 'transcription']):
        return ('main', 'NLP Pipeline', '04-nlp-pipeline.md')
    elif any(keyword in title_lower for keyword in ['expert', 'rules', 'engine', 'reglas', 'rule']):
        return ('main', 'Expert System', '05-expert-system.md')
    elif any(keyword in title_lower for keyword in ['developer', 'api', 'integration', 'guide', 'dev']):
        return ('main', 'Developer Guide', '06-developer-guide.md')

    # Check content for additional context
    elif any(keyword in content_lower for keyword in ['streamlit', 'dashboard', 'st.']) and 'strategy' in content_lower:
        return ('main', 'Streamlit Dashboard', '02-streamlit-dashboard.md')
    elif any(keyword in content_lower for keyword in ['machine learning', 'sklearn', 'tensorflow', 'pytorch']):
        return ('main', 'Machine Learning Models', '03-machine-learning-models.md')
    elif any(keyword in content_lower for keyword in ['nlp', 'sentiment', 'transcription', 'speech']):
        return ('main', 'NLP Pipeline', '04-nlp-pipeline.md')

    # Fallback - categorize as general documentation
    safe_name = safe_filename(title, fallback_idx)
    return ('misc', 'Other', f'99-{safe_name}.md')


def extract_and_download_images(html_content, base_url, image_dir):
    """Extract image URLs from HTML and download them"""
    downloaded_images = {}

    # Find all img tags with various attribute patterns
    img_patterns = [
        r'<img\s+[^>]*src=["\'](.*?)["\'][^>]*>',
        r'<img\s+[^>]*src=([^\s>]+)[^>]*>',
        r'!\[([^\]]*)\]\(([^)]+)\)'  # Markdown images
    ]

    images_found = []

    for pattern in img_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        if pattern.startswith('!'):  # Markdown pattern
            images_found.extend([(m[1], m[0]) for m in matches])
        else:
            # Extract alt text if available
            for match in matches:
                alt_match = re.search(
                    r'alt=["\'](.*?)["\']',
                    html_content[max(0, html_content.find(
                        match)-100):html_content.find(match)+100]
                )
                alt_text = alt_match.group(1) if alt_match else ''
                images_found.append((match, alt_text))

    downloaded_paths = {}

    for img_url, alt_text in images_found:
        # Skip if already downloaded
        if img_url in downloaded_images:
            downloaded_paths[img_url] = downloaded_images[img_url]
            continue

        try:
            # Handle relative and absolute URLs
            if img_url.startswith('http://') or img_url.startswith('https://'):
                full_url = img_url
            elif img_url.startswith('//'):
                full_url = 'https:' + img_url
            elif img_url.startswith('/'):
                # Extract base domain from base_url
                parsed = urllib.parse.urlparse(base_url)
                full_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"
            else:
                # Relative URL
                full_url = urllib.parse.urljoin(base_url, img_url)

            # Generate local filename
            url_path = urllib.parse.urlparse(full_url).path
            if url_path:
                filename = os.path.basename(url_path)
                if not filename or filename == '/':
                    filename = f"image_{len(downloaded_images)}.png"
            else:
                filename = f"image_{len(downloaded_images)}.png"

            # Ensure unique filename
            base_name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(os.path.join(image_dir, filename)):
                filename = f"{base_name}_{counter}{ext}"
                counter += 1

            local_path = os.path.join(image_dir, filename)

            # Download image using curl
            print(f"Downloading image: {full_url} -> {local_path}")
            result = subprocess.run(
                ['curl', '-L', '-s', '-o', local_path,
                    '--create-dirs', '--max-time', '30', full_url],
                capture_output=True
            )

            if result.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                # Store relative path for markdown
                relative_path = os.path.relpath(
                    local_path, os.path.dirname(image_dir))
                downloaded_images[img_url] = relative_path
                downloaded_paths[img_url] = (relative_path, alt_text)
                print(f"✅ Downloaded: {filename}")
            else:
                print(f"❌ Failed to download: {full_url}")
                downloaded_paths[img_url] = (None, alt_text)
                # Clean up empty file if created
                if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
                    os.remove(local_path)

        except Exception as e:
            print(f"❌ Error downloading {img_url}: {e}")
            downloaded_paths[img_url] = (None, alt_text)

    return downloaded_paths


def convert_html_to_markdown_with_images(html_content, image_paths):
    """Convert HTML to Markdown while preserving image positions"""
    content = html_content

    # First, handle img tags
    for img_url, (local_path, alt_text) in image_paths.items():
        if local_path:
            # Replace HTML img with Markdown image
            markdown_img = f"![{alt_text or 'Image'}]({local_path})"
        else:
            # Image couldn't be downloaded
            markdown_img = f"[IMAGE NOT AVAILABLE: {os.path.basename(img_url) or 'unknown'}]"

        # Replace various img tag patterns
        patterns = [
            rf'<img\s+[^>]*src=["\']?{re.escape(img_url)}["\']?[^>]*>',
            rf'!\[[^\]]*\]\({re.escape(img_url)}\)'
        ]

        for pattern in patterns:
            content = re.sub(pattern, markdown_img,
                             content, flags=re.IGNORECASE)

    return content
