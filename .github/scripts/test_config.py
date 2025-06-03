#!/usr/bin/env python3
"""
Script de prueba para verificar la configuración de DeepWiki MCP
"""

import os
import sys
import json
import requests
from wiki_utils import clean_deepwiki_content, safe_filename, categorize_section


def test_environment():
    """Test environment variables"""
    print("🔍 Testing environment variables...")

    required_vars = ['DEEPWIKI_URL', 'DEEPWIKI_MCP_URL',
                     'GITHUB_TOKEN', 'GITHUB_REPOSITORY', 'GITHUB_WORKSPACE']
    missing_vars = []

    for var in required_vars:
        value = os.environ.get(var, '')
        if value:
            print(f"✅ {var}: {value[:50]}{'...' if len(value) > 50 else ''}")
        else:
            print(f"❌ {var}: NOT SET")
            missing_vars.append(var)

    return len(missing_vars) == 0


def test_mcp_connection():
    """Test MCP server connection"""
    print("🔍 Testing MCP server connection...")

    mcp_url = os.environ.get('DEEPWIKI_MCP_URL', 'http://localhost:3000/mcp')

    try:
        # Test simple HTTP connection first
        response = requests.get(mcp_url, timeout=5)
        print(f"✅ MCP server responded with status: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to MCP server at: {mcp_url}")
        print("   Make sure the server is running!")
        return False
    except Exception as e:
        print(f"❌ Error connecting to MCP server: {e}")
        return False


def test_utility_functions():
    """Test utility functions"""
    print("🔍 Testing utility functions...")

    # Test content cleaning
    test_content = """
    Menu
    - Overview
    - System Architecture
    - Streamlit Dashboard
    
    # Test Section
    
    This is some test content with navigation menu above.
    """

    cleaned = clean_deepwiki_content(test_content)
    print(f"✅ Content cleaning: {len(cleaned)} chars after cleaning")

    # Test filename generation
    test_title = "System Architecture & Design"
    filename = safe_filename(test_title)
    print(f"✅ Filename generation: '{test_title}' -> '{filename}'")

    # Test categorization
    category_info = categorize_section(test_title)
    print(f"✅ Categorization: '{test_title}' -> {category_info}")

    return True


def test_json_rpc_payload():
    """Test JSON-RPC payload structure"""
    print("🔍 Testing JSON-RPC payload...")

    deepwiki_url = os.environ.get('DEEPWIKI_URL', 'https://example.com')

    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "tool": "read_wiki_contents",
            "options": {
                "url": deepwiki_url,
                "maxDepth": 1,
                "mode": "pages"
            }
        },
        "id": 1
    }

    try:
        json_str = json.dumps(payload, indent=2)
        print(f"✅ JSON-RPC payload is valid JSON:")
        print(json_str[:200] + "..." if len(json_str) > 200 else json_str)
        return True
    except Exception as e:
        print(f"❌ Invalid JSON-RPC payload: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting DeepWiki MCP configuration tests...\n")

    tests = [
        ("Environment Variables", test_environment),
        ("Utility Functions", test_utility_functions),
        ("JSON-RPC Payload", test_json_rpc_payload),
        ("MCP Connection", test_mcp_connection)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Test: {test_name}")
        print('='*50)

        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print(f"\n{'🎯 SUMMARY':=^50}")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Configuration looks good.")
        return True
    else:
        print(f"⚠️ {total - passed} test(s) failed. Check configuration.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
