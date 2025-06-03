# DeepWiki MCP GitHub Actions Workflow

## 📋 Resumen

Este workflow automatiza la generación de documentación wiki desde contenido DeepWiki utilizando el protocolo Model Context Protocol (MCP). El sistema descarga contenido, procesa imágenes, filtra contenido privado y genera archivos Markdown listos para GitHub Wiki.

## 🏗️ Arquitectura

```
GitHub Actions Workflow
├── 🐍 Setup Python + Node.js
├── 🔧 DeepWiki MCP Server Setup
├── 📡 JSON-RPC Content Extraction
├── 🖼️ Image Download & Processing
├── 🧹 Content Cleaning & Filtering
├── 📝 Markdown Generation
└── 📚 Wiki Publication
```

## 📁 Estructura de Archivos

```
.github/
├── workflows/
│   └── update-wiki.yml              # Main workflow file
└── scripts/
    ├── process_wiki.py              # Main processing script
    ├── wiki_utils.py                # Utility functions
    └── test_config.py               # Configuration test script
```

## 🔧 Componentes Principales

### 1. Workflow Principal (`update-wiki.yml`)

- **Triggers**: Cron diario (6:00 AM UTC) + manual
- **Environment**: Ubuntu latest con Python 3.9 y Node.js 18
- **MCP Server**: Clona, compila y ejecuta servidor DeepWiki
- **Processing**: Ejecuta scripts Python externos
- **Cleanup**: Detiene servidor MCP al finalizar

### 2. Script de Procesamiento (`process_wiki.py`)

- **MCP Integration**: Protocolo JSON-RPC 2.0 completo
- **Content Extraction**: Extrae páginas usando herramienta `read_wiki_contents`
- **Image Processing**: Descarga imágenes con curl y las convierte a Markdown
- **Content Filtering**: Excluye repositorios privados ("99 private repo", etc.)
- **File Generation**: Crea archivos MD categorizados + índice

### 3. Utilidades (`wiki_utils.py`)

- **Content Cleaning**: Remueve menús de navegación y elementos UI
- **Filename Safety**: Genera nombres de archivo seguros
- **Categorization**: Organiza contenido por categorías
- **Image Handling**: Manejo completo de imágenes HTML/Markdown

## 🎯 Características Implementadas

### ✅ Protocolo DeepWiki MCP

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "tool": "read_wiki_contents",
    "options": {
      "url": "<DEEPWIKI_URL>",
      "maxDepth": 1,
      "mode": "pages"
    }
  },
  "id": 1
}
```

### ✅ Procesamiento de Imágenes

- **HTML**: `<img src="/assets/logo.png" alt="System Logo">`
- **Markdown**: `![System Logo](images/assets/logo.png)`
- **Fallback**: `[IMAGE NOT AVAILABLE: logo.png]`

### ✅ Filtrado de Contenido

- Exclusión de "99 private repo" (case-insensitive)
- Filtros adicionales: "private repo", "access denied", etc.
- Verificación tanto en títulos como en contenido

### ✅ Limpieza de Navegación

Patrones regex mejorados para remover menús de navegación en:

- Archivos numerados dobles (01-01)
- Archivos numerados simples (02-)
- Archivos sin numeración
- Menús de bullets y "On this page"

### ✅ Límites de Caracteres

- Código separado en archivos externos
- Workflow principal <2000 caracteres
- Evita el límite de 21000 caracteres de GitHub

## 🚀 Uso

### Ejecución Manual

```bash
# En GitHub Actions
workflow_dispatch: manual trigger

# Local testing (requiere MCP server)
cd .github/scripts
python test_config.py      # Test configuration
python process_wiki.py     # Run processing
```

### Variables de Entorno

```yaml
DEEPWIKI_URL: "https://deepwiki.com/u/VforVitorio/F1_Strat_Manager"
DEEPWIKI_MCP_PORT: "3000"
DEEPWIKI_MCP_URL: "http://localhost:3000/mcp"
DOCS_DIR: "./wiki-docs"
IMAGE_DIR: "./wiki-docs/images"
```

## 📊 Salida Esperada

```
wiki-docs/
├── Home.md                          # Índice principal
├── main-overview.md                 # Documentación principal
├── main-streamlit-dashboard.md      # Dashboard
├── main-machine-learning-models.md # ML Models
├── main-nlp-pipeline.md            # NLP Pipeline
├── main-expert-system.md           # Expert System
├── main-developer-guide.md         # Developer Guide
└── images/                         # Imágenes descargadas
    └── assets/
        └── logo.png
```

## 🧪 Testing

```bash
# Test configuration
python .github/scripts/test_config.py

# Expected output:
# ✅ Environment Variables: PASSED
# ✅ Utility Functions: PASSED
# ✅ JSON-RPC Payload: PASSED
# ❌ MCP Connection: FAILED (expected without server)
```

## 🔍 Troubleshooting

| Issue                | Solution                                           |
| -------------------- | -------------------------------------------------- |
| MCP server timeout   | Aumentar timeout en workflow (actualmente 30s)     |
| Image download fails | Verificar permisos y conectividad de red           |
| Content not filtered | Revisar patrones regex en `clean_deepwiki_content` |
| Wiki push fails      | Verificar `GITHUB_TOKEN` permissions               |

## 📈 Métricas de Éxito

- ✅ Workflow <21000 caracteres
- ✅ Protocolo MCP JSON-RPC implementado
- ✅ Imágenes descargadas y convertidas
- ✅ Contenido privado filtrado
- ✅ Menús de navegación removidos
- ✅ Archivos categorizados correctamente

---

**📝 Última actualización**: Implementación completa con todos los requisitos técnicos  
**🔄 Status**: Ready for production testing  
**🎯 Next**: Testing completo con servidor MCP real
