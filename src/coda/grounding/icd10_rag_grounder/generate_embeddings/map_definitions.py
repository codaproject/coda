"""Map ICD-10 codes to definitions with synonyms."""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import re
import json
import csv

from .config import DEFAULT_ICD10_ZIP, DEFAULT_MRCONSO_PATH, DEFAULT_MRDEF_PATH, SOURCE_PRIORITY


def is_valid_diagnosis_code(code):
    """Check if code is a valid diagnosis code."""
    if len(code) <= 2 or '-' in code or code.endswith(':'):
        return False
    return bool(re.match(r'^[A-Z]\d{2}(\.\d+)?$', code))


def extract_icd10_codes(icd10_zip_path, verbose=True):
    """Extract ICD-10 codes from XML zip file."""
    if verbose:
        print("Extracting ICD-10 codes from XML...")
    
    icd10_codes = {}
    with zipfile.ZipFile(icd10_zip_path, 'r') as zf:
        with zf.open('icd102019en.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for class_elem in root.iter('Class'):
                code = class_elem.get('code')
                if code:
                    name = None
                    for rubric in class_elem.findall('.//Rubric'):
                        label = rubric.find('Label')
                        if label is not None:
                            name = label.text
                            break
                    if name:
                        icd10_codes[code] = name
    
    valid_codes = {code: name for code, name in icd10_codes.items() if is_valid_diagnosis_code(code)}
    
    if verbose:
        print(f"  ✓ Extracted {len(valid_codes)} valid ICD-10 diagnosis codes")
    
    return valid_codes


def collect_strings_from_mrconso(mrconso_path, valid_codes, verbose=True):
    """Collect all strings/synonyms for ICD-10 codes from MRCONSO.RRF."""
    if verbose:
        print("Collecting all strings/synonyms from MRCONSO.RRF...")
    
    code_to_cuis = defaultdict(set)
    code_to_strings = defaultdict(list)
    
    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
        line_count = 0
        for line in f:
            line_count += 1
            if verbose and line_count % 2000000 == 0:
                print(f"  Processed {line_count:,} lines...")
            
            parts = line.strip().split('|')
            if len(parts) >= 15:
                sab = parts[11]
                if sab in ('ICD10', 'ICD10CM', 'ICD10AM'):
                    cui = parts[0]
                    code = parts[13]
                    string = parts[14]
                    ispref = parts[6] if len(parts) > 6 else ''
                    tty = parts[12] if len(parts) > 12 else ''
                    
                    if code in valid_codes:
                        code_to_cuis[code].add(cui)
                        code_to_strings[code].append({
                            'string': string,
                            'ispref': ispref == 'Y',
                            'tty': tty,
                            'sab': sab
                        })
    
    if verbose:
        print(f"  ✓ Found strings for {len(code_to_strings)} codes")
        print(f"  ✓ Codes with multiple strings: {sum(1 for s in code_to_strings.values() if len(s) > 1):,}")
    
    return code_to_cuis, code_to_strings


def load_definitions_from_mrdef(mrdef_path, all_cuis, verbose=True):
    """Load definitions from MRDEF.RRF for given CUIs."""
    if verbose:
        print("Loading definitions from MRDEF.RRF...")
    
    cui_to_definitions = defaultdict(list)
    
    with open(mrdef_path, 'r', encoding='utf-8', errors='ignore') as f:
        line_count = 0
        for line in f:
            line_count += 1
            if verbose and line_count % 100000 == 0:
                print(f"  Processed {line_count:,} lines, found {len(cui_to_definitions):,} CUIs...")
            
            parts = line.strip().split('|')
            if len(parts) >= 6:
                cui = parts[0]
                if cui in all_cuis:
                    sab = parts[4] if len(parts) > 4 else ''
                    definition = parts[5] if len(parts) > 5 else ''
                    if definition and definition.strip():
                        cui_to_definitions[cui].append((sab, definition))
    
    if verbose:
        print(f"  ✓ Found definitions for {len(cui_to_definitions)} CUIs")
    
    return cui_to_definitions


def get_best_definition(cuis, cui_to_definitions, source_priority=SOURCE_PRIORITY):
    """Get best definition from CUIs based on source priority."""
    best_def = None
    best_source = None
    best_priority = float('inf')
    
    for cui in cuis:
        if cui in cui_to_definitions:
            for sab, definition in cui_to_definitions[cui]:
                priority = len(source_priority)
                if sab in source_priority:
                    priority = source_priority.index(sab)
                if priority < best_priority:
                    best_priority = priority
                    best_def = definition
                    best_source = sab
    
    return best_def, best_source


def combine_strings_and_definition(strings, definition):
    """Combine synonyms with definition for richer text."""
    # Get unique strings (synonyms)
    unique_strings = []
    seen = set()
    for s in strings:
        string_lower = s['string'].lower().strip()
        if string_lower not in seen:
            seen.add(string_lower)
            unique_strings.append(s['string'])
    
    # Combine: synonyms + definition
    if len(unique_strings) > 1:
        # Multiple synonyms - combine them
        synonyms_text = "; ".join(unique_strings[:5])  # Limit to 5 synonyms
        if definition:
            combined = f"{synonyms_text}. {definition}"
        else:
            combined = synonyms_text
    else:
        # Single string, use with definition
        if definition:
            combined = f"{unique_strings[0]}. {definition}"
        else:
            combined = unique_strings[0] if unique_strings else ""
    
    return combined


def map_icd10_to_definitions(
    icd10_zip_path=None,
    mrconso_path=None,
    mrdef_path=None,
    output_json=None,
    output_csv=None,
    verbose=True
):
    """
    Map ICD-10 codes to definitions with synonyms.
    
    Args:
        icd10_zip_path: Path to ICD-10 XML zip file
        mrconso_path: Path to MRCONSO.RRF file
        mrdef_path: Path to MRDEF.RRF file
        output_json: Output JSON file path (default: icd10_code_to_definition.json)
        output_csv: Output CSV file path (default: icd10_code_to_definition.csv)
        verbose: Print progress messages
    
    Returns:
        Dictionary mapping codes to definition data
    """
    # Use defaults if not provided
    icd10_zip_path = icd10_zip_path or DEFAULT_ICD10_ZIP
    mrconso_path = mrconso_path or DEFAULT_MRCONSO_PATH
    mrdef_path = mrdef_path or DEFAULT_MRDEF_PATH
    output_json = output_json or Path('icd10_code_to_definition.json')
    output_csv = output_csv or Path('icd10_code_to_definition.csv')
    
    if verbose:
        print("=" * 70)
        print("ICD-10 Code to Definition Mapping")
        print("=" * 70)
    
    # Step 1: Extract ICD-10 codes
    valid_codes = extract_icd10_codes(icd10_zip_path, verbose=verbose)
    
    # Step 2: Collect strings from MRCONSO
    code_to_cuis, code_to_strings = collect_strings_from_mrconso(
        mrconso_path, valid_codes, verbose=verbose
    )
    
    # Step 3: Get all CUIs
    all_cuis = set()
    for cuis in code_to_cuis.values():
        all_cuis.update(cuis)
    
    # Step 4: Load definitions
    cui_to_definitions = load_definitions_from_mrdef(
        mrdef_path, all_cuis, verbose=verbose
    )
    
    # Step 5: Create mappings
    if verbose:
        print("\nCreating mappings...")
        print("-" * 70)
    
    icd10_data = {}
    
    for code in valid_codes.keys():
        name = valid_codes[code]
        strings = code_to_strings.get(code, [])
        cuis = code_to_cuis.get(code, set())
        
        # Get definition
        definition, def_source = get_best_definition(cuis, cui_to_definitions) if cuis else (None, None)
        
        # Combine strings and definition
        if strings:
            combined_text = combine_strings_and_definition(strings, definition)
        else:
            combined_text = definition if definition else name
        
        # Fallback to name if nothing else
        if not combined_text:
            combined_text = name
        
        icd10_data[code] = {
            'code': code,
            'name': name,
            'definition': combined_text,
            'source': def_source if definition else 'ICD10_XML',
            'has_definition': definition is not None,
            'num_cuis': len(cuis),
            'num_strings': len(strings),
            'synonyms': [s['string'] for s in strings[:10]],
            'original_definition': definition
        }
    
    # Statistics
    codes_with_defs = sum(1 for v in icd10_data.values() if v['has_definition'])
    codes_with_synonyms = sum(1 for v in icd10_data.values() if v['num_strings'] > 1)
    
    if verbose:
        print(f"\nResults:")
        print("-" * 70)
        print(f"  Total codes: {len(icd10_data):,}")
        print(f"  Codes with UMLS definitions: {codes_with_defs:,} ({codes_with_defs/len(icd10_data)*100:.1f}%)")
        print(f"  Codes with multiple strings/synonyms: {codes_with_synonyms:,} ({codes_with_synonyms/len(icd10_data)*100:.1f}%)")
        
        avg_length = sum(len(v['definition']) for v in icd10_data.values()) / len(icd10_data)
        print(f"  Average definition length: {avg_length:.1f} chars")
    
    # Save JSON
    if verbose:
        print(f"\nSaving data to {output_json}...")
        print("-" * 70)
    
    output_json = Path(output_json)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(icd10_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"  ✓ Saved {len(icd10_data):,} mappings")
    
    # Save CSV
    output_csv = Path(output_csv)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['code', 'name', 'definition', 'source', 'has_definition', 'num_synonyms'])
        for code, data in sorted(icd10_data.items()):
            writer.writerow([
                data['code'],
                data['name'],
                data['definition'],
                data['source'],
                data['has_definition'],
                data['num_strings']
            ])
    
    if verbose:
        print(f"  ✓ Saved CSV: {output_csv}")
        print("\n" + "=" * 70)
        print("✓ Mapping complete!")
        print("=" * 70)
    
    return icd10_data

