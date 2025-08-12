from __future__ import annotations
from pathlib import Path
import json, time, sys

STATE = Path(__file__).with_name("state")
DRY_RUN = "--dry-run" in sys.argv

def backup(p: Path):
	"""Write a timestamped .bak alongside the target file."""
	bak = p.parent / ".bak"
	bak.mkdir(parents=True, exist_ok=True)
	ts = time.strftime("%Y%m%d-%H%M%S")
	(bak / f"{p.name}.{ts}.jsonbak").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

def set_block_t(block_file: Path, letter: str) -> bool:
	"""Ensure top-level 't' == letter and each grid cell has 't' if missing."""
	if not block_file.exists():
		return False
	changed = False
	try:
		data = json.loads(block_file.read_text(encoding="utf-8"))
	except Exception as e:
		print(f"[WARN] Cannot parse {block_file}: {e}")
		return False

	# top-level t
	if data.get("t") != letter:
		data["t"] = letter
		changed = True

	# per-cell t (grid or cells)
	grid = data.get("grid") or data.get("cells")
	if isinstance(grid, list):
		for r, row in enumerate(grid):
			if not isinstance(row, list):
				continue
			for c, cell in enumerate(row):
				if isinstance(cell, dict) and "t" not in cell:
					cell["t"] = letter
					changed = True

	if changed and not DRY_RUN:
		backup(block_file)
		block_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
	return changed

def ensure_atom_code(bits_file: Path) -> bool:
	"""Ensure Atoms have a 'code' container (supports bits.json or atoms.json)."""
	if not bits_file.exists():
		return False
	try:
		data = json.loads(bits_file.read_text(encoding="utf-8"))
	except Exception as e:
		print(f"[WARN] Cannot parse {bits_file}: {e}")
		return False

	changed = False
	if "code" not in data or not isinstance(data["code"], dict):
		data["code"] = {"entries": {}, "meta": {"created_by": "upgrade_state_min", "ts": time.time()}}
		changed = True

	if changed and not DRY_RUN:
		backup(bits_file)
		bits_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
	return changed

def main():
	if not STATE.exists():
		print("No 'state' folder next to this script.")
		return

	blocks_dir = STATE / "Quarks" / "Blocks"
	atoms_bits = STATE / "Atoms" / "bits.json"
	atoms_json = STATE / "Atoms" / "atoms.json"

	mapping = {
		"data.json": "D",
		"resources.json": "R",
		"applications.json": "A",
		"management.json": "M",
	}

	changed_blocks = 0
	for fname, letter in mapping.items():
		p = blocks_dir / fname
		if set_block_t(p, letter):
			print(f"[OK] Updated '{fname}' -> t='{letter}' (+cells if missing)")
			changed_blocks += 1

	changed_atoms = False
	for af in (atoms_bits, atoms_json):
		if ensure_atom_code(af):
			print(f"[OK] Atom code container ensured in: {af}")
			changed_atoms = True

	print(f"\nSummary: Blocks updated: {changed_blocks}, Atom code added: {bool(changed_atoms)}")
	if not DRY_RUN:
		print("Backups: state/**/.bak/*.jsonbak")
	else:
		print("Dry run only (no files written). Use without '--dry-run' to apply changes.")

if __name__ == "__main__":
	print("DRAM9 upgrade_state_min — add D/R/A/M 't' and Atom code container")
	main()
