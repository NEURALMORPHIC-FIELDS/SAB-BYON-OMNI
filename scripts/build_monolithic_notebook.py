#!/usr/bin/env python3
"""Build the monolithic Colab notebook cell.

Reads all source files and produces a single notebook cell
that writes the entire project to disk using write_source().
Also updates version references from v2.0 to v2.1.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, 'colab', 'SAB_BYON_OMNI_v2.ipynb')
SOURCE_ROOT = os.path.join(PROJECT_ROOT, 'sab_byon_omni')

# Define all source files in dependency order
SOURCE_FILES = [
    # Config
    'sab_byon_omni/config.py',
    # Quantifiers
    'sab_byon_omni/quantifiers/quantification_result.py',
    'sab_byon_omni/quantifiers/base_quantifier.py',
    'sab_byon_omni/quantifiers/statistical_quantifier.py',
    'sab_byon_omni/quantifiers/entropy_quantifier.py',
    'sab_byon_omni/quantifiers/cryptographic_prng.py',
    'sab_byon_omni/quantifiers/reasoning_quantifier.py',
    'sab_byon_omni/quantifiers/memory_relevance_quantifier.py',
    'sab_byon_omni/quantifiers/decision_confidence_quantifier.py',
    'sab_byon_omni/quantifiers/__init__.py',
    # Evolution
    'sab_byon_omni/evolution/metrics_module.py',
    'sab_byon_omni/evolution/frag_param.py',
    'sab_byon_omni/evolution/pathway_evolution.py',
    'sab_byon_omni/evolution/dim1_universal.py',
    'sab_byon_omni/evolution/__init__.py',
    # Memory
    'sab_byon_omni/memory/memory_chunk.py',
    'sab_byon_omni/memory/fragmergent_memory.py',
    'sab_byon_omni/memory/holographic_memory.py',
    'sab_byon_omni/memory/conversation_manager.py',
    'sab_byon_omni/memory/fhrss_fcpe_engine.py',
    'sab_byon_omni/memory/infinite_context.py',
    'sab_byon_omni/memory/__init__.py',
    # Agents
    'sab_byon_omni/agents/base_agent.py',
    'sab_byon_omni/agents/rl_agent.py',
    'sab_byon_omni/agents/fragmergent_agent.py',
    'sab_byon_omni/agents/memory_agent.py',
    'sab_byon_omni/agents/multi_agent_cortex.py',
    'sab_byon_omni/agents/__init__.py',
    # Cognitive
    'sab_byon_omni/cognitive/fisher_geometry.py',
    'sab_byon_omni/cognitive/info_density_field.py',
    'sab_byon_omni/cognitive/semantic_photon.py',
    'sab_byon_omni/cognitive/duei_framework.py',
    'sab_byon_omni/cognitive/personality.py',
    'sab_byon_omni/cognitive/__init__.py',
    # Consciousness
    'sab_byon_omni/consciousness/triadic_state.py',
    'sab_byon_omni/consciousness/tdfc_engine.py',
    'sab_byon_omni/consciousness/godel_engine.py',
    'sab_byon_omni/consciousness/icf.py',
    'sab_byon_omni/consciousness/fragmergent_engine.py',
    'sab_byon_omni/consciousness/time_emergence.py',
    'sab_byon_omni/consciousness/zeta_resonance.py',
    'sab_byon_omni/consciousness/emergence_detector.py',
    'sab_byon_omni/consciousness/__init__.py',
    # Model
    'sab_byon_omni/model/config.py',
    'sab_byon_omni/model/omni_agi_nexus.py',
    'sab_byon_omni/model/__init__.py',
    # Training
    'sab_byon_omni/training/train_3b.py',
    'sab_byon_omni/training/__init__.py',
    # Core
    'sab_byon_omni/core/sab_transcendent.py',
    'sab_byon_omni/core/__init__.py',
    # Top-level __init__
    'sab_byon_omni/__init__.py',
]


def escape_for_triple_single_quote(content):
    """Escape content to safely embed in r'''...''' string literal."""
    # Replace any occurrence of ''' with something safe
    # We use string concatenation to break the triple-quote
    content = content.replace("'''", "'" + "''" )
    return content


def build_monolithic_cell():
    """Build the complete monolithic cell source."""
    lines = []
    lines.append('# ' + '='*70)
    lines.append('# SECTION 3: MONOLITHIC SOURCE CODE - SAB + BYON-OMNI v2.1')
    lines.append('# All 43 capabilities - 50 source files written automatically')
    lines.append('# ' + '='*70)
    lines.append('')
    lines.append('import os, sys')
    lines.append("PROJECT_ROOT = '/content/SAB-BYON-OMNI'")
    lines.append('')
    lines.append('def write_source(relative_path, code):')
    lines.append('    """Write source code to the project tree."""')
    lines.append('    full_path = os.path.join(PROJECT_ROOT, relative_path)')
    lines.append('    os.makedirs(os.path.dirname(full_path), exist_ok=True)')
    lines.append("    with open(full_path, 'w', encoding='utf-8') as f:")
    lines.append('        f.write(code)')
    lines.append("    lines_count = code.count('\\n') + 1")
    lines.append("    print(f'  [WRITTEN] {relative_path} ({lines_count} lines)')")
    lines.append('')
    lines.append("print('='*70)")
    lines.append("print('SAB + BYON-OMNI v2.1 - WRITING ALL SOURCE FILES')")
    lines.append("print('  43 Capabilities | 50 Source Files | Monolithic Deploy')")
    lines.append("print('='*70)")
    lines.append("print()")
    lines.append('')

    total_files = 0
    total_lines = 0

    for rel_path in SOURCE_FILES:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"WARNING: {full_path} not found, skipping")
            continue

        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove trailing whitespace/newlines
        content = content.rstrip()

        # Count lines
        file_lines = content.count('\n') + 1
        total_lines += file_lines
        total_files += 1

        # Escape for embedding
        escaped_content = escape_for_triple_single_quote(content)

        # Add section comment for readability
        module_name = rel_path.replace('sab_byon_omni/', '').replace('/', '.')
        lines.append(f"# --- {module_name} ({file_lines} lines) ---")
        lines.append(f"write_source('{rel_path}', r'''")
        lines.append(escaped_content)
        lines.append("''')")
        lines.append('')

    # Final verification
    lines.append('')
    lines.append("print()")
    lines.append("print('='*70)")
    lines.append(f"print('SOURCE CODE COMPLETE: {total_files} files, ~{total_lines} lines')")
    lines.append("print('  SAB + BYON-OMNI v2.1 - 43 Capabilities')")
    lines.append("print('  FHRSS: Patent EP25216372.0 - 9 parity families')")
    lines.append("print('  FCPE: 73,000x compression - 384-dim embeddings')")
    lines.append("print('  InfiniteContext: 2M+ tokens - SSD persistence')")
    lines.append("print('='*70)")
    lines.append('')
    lines.append("# Add project to Python path")
    lines.append("if PROJECT_ROOT not in sys.path:")
    lines.append("    sys.path.insert(0, PROJECT_ROOT)")
    lines.append("print('\\n[OK] All source files written. Project in sys.path.')")

    return '\n'.join(lines), total_files, total_lines


def update_notebook():
    """Update the Colab notebook with monolithic cell and version bumps."""
    # Read notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # Build monolithic cell content
    cell_source, total_files, total_lines = build_monolithic_cell()
    print(f"Built monolithic cell: {total_files} files, {total_lines} lines")

    # Find cell indices by cell ID
    cell_id_to_idx = {}
    for i, cell in enumerate(cells):
        cell_id = cell.get('metadata', {}).get('id', cell.get('id', ''))
        # Also check source for cell ID pattern
        source = ''.join(cell.get('source', []))
        if 'cell-' in str(cell.get('metadata', {}).get('id', '')):
            cell_id_to_idx[cell['metadata']['id']] = i
        elif 'id' in cell:
            cell_id_to_idx[cell['id']] = i

    # Update cell 0 (markdown header) - v2.0 -> v2.1, 40 -> 43
    for i, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if 'SAB + BYON-OMNI v2.0' in source and cell.get('cell_type') == 'markdown':
            new_source = source.replace('SAB + BYON-OMNI v2.0', 'SAB + BYON-OMNI v2.1')
            new_source = new_source.replace('**40 Capabilities**', '**43 Capabilities**')
            cells[i]['source'] = [new_source]
            print(f"Updated cell {i}: header v2.0 -> v2.1")
            break

    # Update cell 1 (logo display) - v2.0 -> v2.1, 40 -> 43
    for i, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if 'SAB-BYON-OMNI-AI v2.0' in source and 'logo' in source.lower():
            new_source = source.replace('SAB-BYON-OMNI-AI v2.0', 'SAB-BYON-OMNI-AI v2.1')
            new_source = new_source.replace('40 Capabilities', '43 Capabilities')
            cells[i]['source'] = [new_source]
            print(f"Updated cell {i}: logo v2.0 -> v2.1")
            break

    # Update Section 3 markdown (cell 6)
    for i, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if 'SECTION 3: Source Code' in source and cell.get('cell_type') == 'markdown':
            cells[i]['source'] = [
                '---\n'
                '## SECTION 3: Source Code (Monolithic)\n'
                '**All 50 source files are written automatically in ONE cell.**\n\n'
                'v2.1: SAB Original (30) + EAG-Core (5) + ICF (5) + FHRSS + FCPE + InfiniteContext = **43 capabilities**'
            ]
            print(f"Updated cell {i}: Section 3 markdown")
            break

    # Create the monolithic code cell
    monolithic_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {
            'id': 'cell-monolithic-source'
        },
        'outputs': [],
        'source': [cell_source]
    }

    # Strategy 1: Replace existing monolithic cell (by metadata ID)
    replaced = False
    for i, cell in enumerate(cells):
        if cell.get('metadata', {}).get('id') == 'cell-monolithic-source':
            cells[i] = monolithic_cell
            print(f"Replaced existing monolithic cell at index {i}")
            replaced = True
            break

    if not replaced:
        # Strategy 2: Find and remove placeholder cells, insert monolithic
        cells_to_remove = []
        monolithic_insert_idx = None

        for i, cell in enumerate(cells):
            source = ''.join(cell.get('source', []))
            if cell.get('cell_type', 'code') == 'code':
                if "def write_source(relative_path, code):" in source and "write_source() helper ready" in source:
                    continue
                if ('--- 3.' in source and 'PASTE' in source.upper()) or \
                   ('Uncomment and paste' in source) or \
                   ("write_source('sab_byon_omni/" in source and 'PASTE' in source.upper()):
                    if monolithic_insert_idx is None:
                        monolithic_insert_idx = i
                    cells_to_remove.append(i)

        for idx in sorted(cells_to_remove, reverse=True):
            cells.pop(idx)

        if monolithic_insert_idx is not None:
            cells.insert(monolithic_insert_idx, monolithic_cell)
            print(f"Inserted monolithic cell at index {monolithic_insert_idx}")
        else:
            print("WARNING: Could not find insertion point")

        # Remove old write_source helper if present
        for i, cell in enumerate(cells):
            source = ''.join(cell.get('source', []))
            if "def write_source(relative_path, code):" in source and "write_source() helper ready" in source:
                cells.pop(i)
                print(f"Removed old write_source helper cell at index {i}")
                break

    # Update training and benchmark cells: v2.0 -> v2.1
    for i, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if 'v2.0' in source:
            new_source = source.replace('v2.0', 'v2.1')
            cells[i]['source'] = [new_source]

    # Update default.yaml in cell 3 from v2.0 to v2.1
    for i, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if "# SAB + BYON-OMNI v2.0 Default Configuration" in source:
            new_source = source.replace(
                "# SAB + BYON-OMNI v2.0 Default Configuration",
                "# SAB + BYON-OMNI v2.1 Default Configuration"
            )
            cells[i]['source'] = [new_source]
            print(f"Updated cell {i}: config yaml v2.0 -> v2.1")

    nb['cells'] = cells

    # Write updated notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\nNotebook updated: {NOTEBOOK_PATH}")
    print(f"Total cells: {len(cells)}")


if __name__ == '__main__':
    update_notebook()
