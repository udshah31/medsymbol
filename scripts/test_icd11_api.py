#!/usr/bin/env python3
"""
ICD-11 API Connectivity Test & Explorer
========================================
Tests connection to the WHO ICD-11 API and explores relevant codes
for the MedSymbol respiratory/thoracic domain.

Usage:
    1. Set your credentials below (client_id and client_secret)
    2. Run: python scripts/test_icd11_api.py
"""

import json
import urllib.request
import urllib.parse

# ═══════════════════════════════════════════════════
# ⚠️  SET YOUR ICD-11 API CREDENTIALS HERE
# ═══════════════════════════════════════════════════
CLIENT_ID = "bc8a27c1-b9c8-46b1-9cf5-3bc890accdfd_7b67ecac-cbc8-4b79-8c78-c3ccbccfc288"
CLIENT_SECRET = "1FHYSrwpzLCsL8V9dszhGx//ECnM0B3W0xCW4s8rdT8="
# ═══════════════════════════════════════════════════

TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
API_BASE = "https://id.who.int/icd"

# Respiratory/thoracic ICD-11 codes relevant to MedSymbol
MEDSYMBOL_CODES = {
    "CA40": "Community-acquired pneumonia",
    "CA40.0": "Viral pneumonia",
    "CA40.1": "Bacterial pneumonia",
    "BB01": "Malignant neoplasms of bronchus or lung",
    "BD1": "Chronic obstructive pulmonary disease",
    "CA20": "Acute bronchitis",
    "CB21": "Asthma",
    "BA00": "Heart failure",
    "BC40": "Cardiomegaly",
    "KB20": "Pleural effusion",
}


def get_token(client_id: str, client_secret: str) -> str:
    """Obtain OAuth2 access token from WHO."""
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "icdapi_access",
    }).encode()

    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        return result["access_token"]


def query_icd11(token: str, code: str) -> dict:
    """Query the ICD-11 API for a specific code."""
    url = f"{API_BASE}/release/11/2024-01/mms/codeinfo/{code}?flexiblemode=true"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("Accept-Language", "en")
    req.add_header("API-Version", "v2")

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def search_icd11(token: str, query: str, max_results: int = 5) -> dict:
    """Search ICD-11 for a text query."""
    encoded_query = urllib.parse.quote(query)
    url = (
        f"{API_BASE}/release/11/2024-01/mms/search"
        f"?q={encoded_query}&subtreeFilterUsesFoundationDescendants=false"
        f"&includeKeywordResult=false&useFlexisearch=true"
        f"&flatResults=true&highlightingEnabled=false"
    )
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("Accept-Language", "en")
    req.add_header("API-Version", "v2")

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def main():
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold blue]═══ ICD-11 API Connectivity Test ═══[/bold blue]\n")

    if CLIENT_ID == "YOUR_CLIENT_ID_HERE":
        console.print("[bold red]❌ Please set your CLIENT_ID and CLIENT_SECRET at the top of this script.[/bold red]")
        console.print("[dim]Get them from: https://icd.who.int/icdapi[/dim]\n")
        return

    # Step 1: Get token
    console.print("[cyan]Step 1:[/cyan] Obtaining OAuth2 token...")
    try:
        token = get_token(CLIENT_ID, CLIENT_SECRET)
        console.print(f"  [green]✓[/green] Token obtained (length: {len(token)})\n")
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed: {e}\n")
        return

    # Step 2: Query known codes
    console.print("[cyan]Step 2:[/cyan] Querying MedSymbol-relevant ICD-11 codes...\n")

    table = Table(title="ICD-11 Codes for MedSymbol", show_header=True, header_style="bold cyan")
    table.add_column("Code", style="bold", width=10)
    table.add_column("Expected", width=30)
    table.add_column("API Title", width=35)
    table.add_column("Status", width=8)

    for code, expected_name in MEDSYMBOL_CODES.items():
        result = query_icd11(token, code)
        if "error" in result:
            table.add_row(code, expected_name, f"Error: {result['error'][:30]}", "[red]✗[/red]")
        else:
            title = result.get("title", {})
            # title can be a string or a dict with @value
            if isinstance(title, dict):
                title_text = title.get("@value", str(title))
            else:
                title_text = str(title)
            table.add_row(code, expected_name, title_text[:35], "[green]✓[/green]")

    console.print(table)

    # Step 3: Test search
    console.print("\n[cyan]Step 3:[/cyan] Testing search for 'pneumonia'...\n")
    results = search_icd11(token, "pneumonia")
    if "error" not in results:
        entities = results.get("destinationEntities", [])[:5]
        for entity in entities:
            title = entity.get("title", "N/A")
            code = entity.get("theCode", "N/A")
            console.print(f"  [green]→[/green] {code}: {title}")
        console.print(f"\n  [dim]Total results: {results.get('resultCount', 'unknown')}[/dim]")
    else:
        console.print(f"  [red]✗[/red] Search failed: {results['error']}")

    # Save relevant codes for later use
    output_path = "data/ontology/icd11_medsymbol_codes.json"
    import os
    os.makedirs("data/ontology", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(MEDSYMBOL_CODES, f, indent=2)
    console.print(f"\n[green]✓[/green] Saved MedSymbol ICD-11 codes to {output_path}")

    console.print("\n[bold green]✅ ICD-11 API is working![/bold green]\n")


if __name__ == "__main__":
    main()
