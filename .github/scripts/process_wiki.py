#!/usr/bin/env python3
"""
Main script for processing DeepWiki content and generating GitHub Wiki files.
This script uses DeepWiki MCP JSON-RPC protocol for proper content extraction.
Handles image downloading and proper Markdown conversion as specified.
"""

import os
import sys
import json
import requests
from bs4 import BeautifulSoup
import traceback
import subprocess
import urllib.parse
import re
from pathlib import Path
from wiki_utils import (
    clean_deepwiki_content,
    get_page_title_from_content,
    safe_filename,
    categorize_section
)

# Configuration from environment variables
DEEPWIKI_URL = os.environ.get('DEEPWIKI_URL', '')
DEEPWIKI_MCP_URL = os.environ.get(
    'DEEPWIKI_MCP_URL', 'http://localhost:3000/mcp')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_REPOSITORY = os.environ.get('GITHUB_REPOSITORY', '')
GITHUB_WORKSPACE = os.environ.get('GITHUB_WORKSPACE', '')

# Directories
DOCS_DIR = os.path.join(GITHUB_WORKSPACE, 'wiki-docs')
IMAGE_DIR = os.path.join(DOCS_DIR, 'images')

# Global image tracking
downloaded_images = {}


def call_deepwiki_mcp(url, max_depth=1, mode="pages"):
    """Call DeepWiki MCP using JSON-RPC protocol"""
    try:
        # Prepare JSON-RPC payload according to specification
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "tool": "read_wiki_contents",
                "options": {
                    "url": url,
                    "maxDepth": max_depth,
                    "mode": mode
                }
            },
            "id": 1
        }

        print(f"🔧 Calling DeepWiki MCP at: {DEEPWIKI_MCP_URL}")
        print(f"📋 Request payload: {json.dumps(payload, indent=2)}")

        # Make the JSON-RPC call
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'GitHub-Actions-Wiki-Bot/1.0'
        }

        response = requests.post(
            DEEPWIKI_MCP_URL,
            json=payload,
            headers=headers,
            timeout=60
        )

        response.raise_for_status()

        # Parse response
        mcp_response = response.json()
        print(f"📥 MCP Response received, keys: {list(mcp_response.keys())}")

        return mcp_response

    except Exception as e:
        print(f"❌ Error calling DeepWiki MCP: {e}")
        traceback.print_exc()
        return None


def extract_pages_from_mcp_response(mcp_response):
    """Extract pages from DeepWiki MCP JSON-RPC response"""
    try:
        if not mcp_response:
            return []

        # Handle JSON-RPC response structure
        if 'result' not in mcp_response:
            if 'error' in mcp_response:
                print(f"❌ MCP Error: {mcp_response['error']}")
            return []

        result = mcp_response['result']
        pages = []

        # Handle different response structures from MCP
        if isinstance(result, dict):
            if 'pages' in result:
                # Multiple pages structure
                pages_data = result['pages']
                if isinstance(pages_data, list):
                    for i, page_data in enumerate(pages_data):
                        if isinstance(page_data, dict) and 'content' in page_data:
                            pages.append({
                                'html': page_data['content'],
                                'url': page_data.get('url', DEEPWIKI_URL),
                                'title': page_data.get('title', f'Page {i+1}'),
                                'id': str(i+1)
                            })
                        elif isinstance(page_data, str):
                            pages.append({
                                'html': page_data,
                                'url': DEEPWIKI_URL,
                                'title': f'Page {i+1}',
                                'id': str(i+1)
                            })
                elif isinstance(pages_data, dict):
                    # Single page in pages structure
                    pages.append({
                        'html': pages_data.get('content', str(pages_data)),
                        'url': pages_data.get('url', DEEPWIKI_URL),
                        'title': pages_data.get('title', 'Documentation'),
                        'id': '1'
                    })
            elif 'content' in result:
                # Direct content structure
                pages.append({
                    'html': result['content'],
                    'url': result.get('url', DEEPWIKI_URL),
                    'title': result.get('title', 'Documentation'),
                    'id': '1'
                })
            else:
                # Fallback: treat entire result as content
                pages.append({
                    'html': str(result),
                    'url': DEEPWIKI_URL,
                    'title': 'Documentation',
                    'id': '1'
                })
        elif isinstance(result, str):
            # Direct string content
            pages.append({
                'html': result,
                'url': DEEPWIKI_URL,
                'title': 'Documentation',
                'id': '1'
            })

        print(f"📄 Extracted {len(pages)} pages from MCP response")
        for page in pages:
            print(
                f"  - Page {page['id']}: {page['title']} ({len(page['html'])} chars)")

        return pages

    except Exception as e:
        print(f"❌ Error extracting pages from MCP response: {e}")
        traceback.print_exc()
        return []


def download_image(img_url, base_url=None):
    """Download a single image and return local path or None if failed"""
    global downloaded_images

    # Skip if already downloaded
    if img_url in downloaded_images:
        return downloaded_images[img_url]

    try:
        # Handle different URL formats
        if img_url.startswith('http://') or img_url.startswith('https://'):
            full_url = img_url
        elif img_url.startswith('//'):
            full_url = 'https:' + img_url
        elif img_url.startswith('/'):
            # Extract base domain from DEEPWIKI_URL or base_url
            parsed = urllib.parse.urlparse(base_url or DEEPWIKI_URL)
            full_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"
        else:
            # Relative URL
            full_url = urllib.parse.urljoin(base_url or DEEPWIKI_URL, img_url)

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
        if not ext:
            ext = '.png'
        counter = 1
        while os.path.exists(os.path.join(IMAGE_DIR, filename)):
            filename = f"{base_name}_{counter}{ext}"
            counter += 1

        local_path = os.path.join(IMAGE_DIR, filename)

        # Download image using curl
        print(f"📸 Downloading image: {full_url} -> {filename}")
        result = subprocess.run(
            ['curl', '-L', '-s', '-o', local_path,
                '--create-dirs', '--max-time', '30', full_url],
            capture_output=True
        )

        if result.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            # Store relative path for markdown (relative to DOCS_DIR)
            relative_path = os.path.relpath(local_path, DOCS_DIR)
            downloaded_images[img_url] = relative_path
            print(f"✅ Downloaded: {filename}")
            return relative_path
        else:
            print(f"❌ Failed to download: {full_url}")
            if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
                os.remove(local_path)
            downloaded_images[img_url] = None
            return None

    except Exception as e:
        print(f"❌ Error downloading {img_url}: {e}")
        downloaded_images[img_url] = None
        return None


def process_html_to_markdown_with_images(html_content, base_url=None):
    """Convert HTML to Markdown while downloading and replacing images"""
    try:
        # Find all img tags with src attributes
        img_pattern = r'<img\s+[^>]*src=["\'](.*?)["\'][^>]*(?:alt=["\'](.*?)["\'])?[^>]*>'

        def replace_img(match):
            img_url = match.group(1)
            alt_text = match.group(2) if match.group(2) else 'Image'

            # Download the image
            local_path = download_image(img_url, base_url)

            if local_path:
                # Return Markdown image syntax
                return f"![{alt_text}]({local_path})"
            else:
                # Image couldn't be downloaded
                filename = os.path.basename(img_url) or 'unknown'
                return f"[IMAGE NOT AVAILABLE: {filename}]"

        # Replace all img tags
        content_with_images = re.sub(
            img_pattern, replace_img, html_content, flags=re.IGNORECASE)

        # Also handle Markdown image syntax that might already exist
        md_img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

        def replace_md_img(match):
            alt_text = match.group(1)
            img_url = match.group(2)

            # Only download if it's not already a local path
            if not img_url.startswith('images/') and not img_url.startswith('./images/'):
                local_path = download_image(img_url, base_url)
                if local_path:
                    return f"![{alt_text}]({local_path})"
                else:
                    filename = os.path.basename(img_url) or 'unknown'
                    return f"[IMAGE NOT AVAILABLE: {filename}]"
            else:
                # Already a local path, keep as is
                return match.group(0)

        content_with_images = re.sub(
            md_img_pattern, replace_md_img, content_with_images)

        return content_with_images

    except Exception as e:
        print(f"❌ Error processing images: {e}")
        return html_content


def process_section_content(page_data):
    """Process individual page content from MCP response"""
    try:
        page_id = page_data['id']
        html_content = page_data['html']
        page_url = page_data['url']
        page_title = page_data['title']

        print(f"🔄 Processing page {page_id}: {page_title}")

        # First, process images in the HTML content
        content_with_images = process_html_to_markdown_with_images(
            html_content, page_url)

        # Clean the content using existing utility
        cleaned_content = clean_deepwiki_content(content_with_images)

        if not cleaned_content or len(cleaned_content.strip()) < 50:
            print(f"⚠️ Page {page_id} has very little content, skipping")
            return None

        # Get title from content or use page info
        title = get_page_title_from_content(cleaned_content)
        if not title or title == "Documentation":
            title = page_title or f'Page {page_id}'

        # Skip files with "99 private repo" in title (case insensitive)
        if "99 private repo" in title.lower():
            print(f"🚫 Skipping page '{title}' (contains '99 private repo')")
            return None

        # Additional checks for private repo content
        private_indicators = [
            'private repo', 'private repository', 'repo private', 'access denied']
        if any(keyword in title.lower() for keyword in private_indicators):
            print(f"🚫 Skipping page '{title}' (private repository content)")
            return None

        # Check content for private repo indicators
        if any(keyword in cleaned_content.lower() for keyword in ['99 private repo', 'private repository', 'access denied']):
            print(
                f"🚫 Skipping page '{title}' (contains private repository content)")
            return None

        # Generate safe filename
        filename = safe_filename(title, page_id)

        # Categorize the section
        category = categorize_section(title, cleaned_content)

        return {
            'id': page_id,
            'title': title,
            'filename': filename,
            'category': category,
            'content': cleaned_content,
            'url': page_url,
            'original_title': page_title
        }

    except Exception as e:
        print(f"❌ Error processing page {page_data.get('id', 'unknown')}: {e}")
        traceback.print_exc()
        return None


def fetch_deepwiki_content():
    """Fetch content from DeepWiki using MCP"""
    try:
        print(f"🔍 Fetching content from DeepWiki: {DEEPWIKI_URL}")

        # Call DeepWiki MCP
        mcp_response = call_deepwiki_mcp(
            DEEPWIKI_URL, max_depth=1, mode="pages")
        if not mcp_response:
            raise Exception("Failed to get response from DeepWiki MCP")

        # Extract pages from response
        pages = extract_pages_from_mcp_response(mcp_response)
        if not pages:
            raise Exception("No pages extracted from MCP response")

        # Process each page
        processed_sections = []
        for page_data in pages:
            processed_section = process_section_content(page_data)
            if processed_section:
                processed_sections.append(processed_section)
                print(f"✅ Processed: {processed_section['title']}")

        return processed_sections

    except Exception as e:
        print(f"❌ Error fetching DeepWiki content: {e}")
        traceback.print_exc()
        return []


def create_directory_structure():
    """Create necessary directories"""
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"📁 Created directories: {DOCS_DIR}")


def save_sections_to_files(sections):
    """Save processed sections to individual files"""
    try:
        if not sections:
            print("⚠️ No sections to save")
            return False

        # Group sections by category
        categories = {}
        for section in sections:
            category = section['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(section)

        # Save sections
        saved_files = []

        for category, category_sections in categories.items():
            print(f"📂 Processing category: {category}")

            for section in category_sections:
                try:
                    # Create filename with category prefix if needed
                    if category != 'general':
                        filename = f"{category.lower().replace(' ', '-')}-{section['filename']}.md"
                    else:
                        filename = f"{section['filename']}.md"

                    filepath = os.path.join(DOCS_DIR, filename)

                    # Prepare content with title
                    content = f"# {section['title']}\n\n{section['content']}"

                    # Write file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)

                    saved_files.append({
                        'filename': filename,
                        'title': section['title'],
                        'category': category,
                        'path': filepath
                    })

                    print(f"💾 Saved: {filename}")

                except Exception as e:
                    print(f"❌ Error saving section {section['title']}: {e}")
                    continue

        # Create index file
        create_index_file(saved_files, categories)

        print(f"✅ Successfully saved {len(saved_files)} files")
        return True

    except Exception as e:
        print(f"❌ Error saving sections: {e}")
        traceback.print_exc()
        return False


def create_index_file(saved_files, categories):
    """Create a main index file with links to all sections"""
    try:
        index_path = os.path.join(DOCS_DIR, 'Home.md')

        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# F1 Strategy Manager - Project Wiki\n\n")
            f.write("Welcome to the F1 Strategy Manager project documentation. This wiki contains comprehensive information about the AI-powered Formula 1 race strategy analysis system.\n\n")

            # Project overview
            f.write("## 🏎️ Project Overview\n\n")
            f.write("The F1 Strategy Manager is an integrated AI-powered system for Formula 1 race strategy analysis and decision support, combining:\n\n")
            f.write(
                "- **🤖 Machine Learning Models** - Predictive analytics for lap times and tire performance\n")
            f.write(
                "- **👁️ Computer Vision** - Automated gap calculation from video feeds\n")
            f.write(
                "- **🎤 Natural Language Processing** - Radio communication analysis and insights\n")
            f.write(
                "- **⚙️ Rule-based Expert Systems** - Strategic recommendations based on F1 expertise\n")
            f.write(
                "- **📊 Interactive Streamlit Dashboard** - User-friendly web interface for real-time analysis\n\n")

            f.write("## 📚 Documentation Sections\n\n")

            # Write category sections
            for category, sections in categories.items():
                if category == 'general':
                    f.write("### General Documentation\n\n")
                else:
                    f.write(f"### {category.title()}\n\n")

                for section in sections:
                    # Find the corresponding saved file
                    saved_file = next(
                        (sf for sf in saved_files if sf['title'] == section['title']), None)
                    if saved_file:
                        # Create wiki link (remove .md extension)
                        link_name = saved_file['filename'].replace('.md', '')
                        f.write(f"- **[{section['title']}]({link_name})**\n")

                f.write("\n")

            # Statistics
            f.write("## 📊 Wiki Statistics\n\n")
            f.write(f"- Total documentation files: {len(saved_files)}\n")
            f.write(f"- Categories: {len(categories)}\n")
            f.write(
                f"- Downloaded images: {len([path for path in downloaded_images.values() if path])}\n")

            f.write("\n---\n\n")
            f.write(
                "*📝 This documentation is automatically generated from the project's DeepWiki content using MCP protocol.*\n")
            f.write("*🔄 Last updated: Automatically via GitHub Actions*\n")
            f.write(
                "*🖼️ Images are automatically downloaded and embedded in the documentation.*\n")

        print("📑 Created index file: Home.md")

    except Exception as e:
        print(f"❌ Error creating index file: {e}")


def main():
    """Main processing function"""
    try:
        print("🚀 Starting DeepWiki MCP processing...")

        # Validate environment
        if not DEEPWIKI_URL:
            raise Exception("DEEPWIKI_URL environment variable is required")

        if not GITHUB_WORKSPACE:
            raise Exception(
                "GITHUB_WORKSPACE environment variable is required")

        # Create directory structure
        create_directory_structure()

        # Fetch and process content using MCP
        sections = fetch_deepwiki_content()

        if not sections:
            print("⚠️ No sections were processed successfully")
            return False

        print(f"📊 Total sections processed: {len(sections)}")

        # Save sections to files
        success = save_sections_to_files(sections)

        if success:
            print("✅ DeepWiki MCP processing completed successfully!")
            print(f"📄 Generated {len(sections)} documentation files")
            print(
                f"🖼️ Downloaded {len([p for p in downloaded_images.values() if p])} images")
            return True
        else:
            print("❌ DeepWiki MCP processing failed")
            return False

    except Exception as e:
        print(f"❌ Fatal error in main processing: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
