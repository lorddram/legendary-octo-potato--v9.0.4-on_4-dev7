# DRAM9_On — On_4: Blocks + Arrow Mode + Wiring + PyENExE hook
# Version: 9.0.4-on_4
#
# Adds to On_3 Final:
# - Display mode toggle: "normal" or "arrows"
# - Block placement/orientation at Synthesis cells (cell.meta.block)
# - Block templates under Quarks/Blocks (kind="block", 3x3 default)
# - Ports: "?" inputs, "~" outputs; rotated with block
# - Wiring commands: wire/unwire + show ports
# - Corner-touch rule (configurable) + auto-guard (replicator)
# - Compile to program via PyENExE
#
# Single-browser/editor paradigm preserved.

from __future__ import annotations
import argparse, json, os, sys, time, re, shutil
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

import pyenexe  # local engine

VERSION = "9.0.4-on_4"
TOPS = ["Quarks", "Atoms", "Synthesis"]
BAK_DIR = ".bak"
LOCK_SUFFIX = ".lock"
BACKUP_KEEP = 5
HISTORY_CAP = 20

DEFAULT_CONFIG = {
    "version": VERSION,
    "symbols": {
        "levels": { "quark": ".", "atom": "^", "synthesis": "#" },
        "cells":  { "D": "D", "R": "R", "A": "A", "M": "M" }
    },
    "grid": { "rows": 18, "cols": 18 },
    "ui": { "display_mode": "normal" },  # normal | arrows
    "blocks": {
        "no_corner_touch": True,
        "replicator_id": "replicator"
    }
}

# --------------------------- File helpers ---------------------------
class FileLock:
    def __init__(self, target: Path, timeout: float = 5.0, poll: float = 0.05):
        self.lock_path = target.with_suffix(target.suffix + LOCK_SUFFIX)
        self.timeout = timeout
        self.poll = poll
    def __enter__(self):
        start = time.time()
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"Lock timeout: {self.lock_path}")
                time.sleep(self.poll)
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass

def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2, ensure_ascii=False)
    with FileLock(path):
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        create_backup(path)

def safe_load_json(path: Path, default=None):
    try:
        with FileLock(path):
            if not path.exists():
                restored = restore_latest_backup(path)
                if restored is not None: return restored
                return default
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except json.JSONDecodeError:
        quarantine_corrupt(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.exists():
            try:
                with open(tmp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                os.replace(tmp, path)
                return obj
            except Exception:
                pass
        restored = restore_latest_backup(path)
        if restored is not None: return restored
        return default

def create_backup(path: Path) -> None:
    bak_root = path.parent / BAK_DIR
    bak_root.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = bak_root / f"{path.name}.{ts}.jsonbak"
    with open(path, "r", encoding="utf-8") as src, open(bak, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    baks = sorted(bak_root.glob(f"{path.name}.*.jsonbak"))
    excess = len(baks) - BACKUP_KEEP
    for p in baks[:max(0, excess)]:
        try: p.unlink()
        except Exception: pass

def restore_latest_backup(path: Path):
    bak_root = path.parent / BAK_DIR
    baks = sorted(bak_root.glob(f"{path.name}.*.jsonbak"))
    for bak in reversed(baks):
        try:
            with open(bak, "r", encoding="utf-8") as f:
                obj = json.load(f)
            atomic_write_json(path, obj)
            return obj
        except Exception:
            continue
    return None

def quarantine_corrupt(path: Path) -> None:
    try:
        data = path.read_bytes()
    except Exception:
        return
    corrupt_dir = path.parent / ".corrupt"
    corrupt_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    (corrupt_dir / f"{path.name}.{ts}.corrupt").write_bytes(data)

# ------------------------------- App plumbing ------------------------------
def resolve_user_root(cli_path: Optional[str]) -> Path:
    if cli_path:
        root = Path(cli_path).expanduser().resolve()
    else:
        envp = os.getenv("DRAM9_USER_STORAGE")
        root = Path(envp).expanduser().resolve() if envp else Path(__file__).resolve().parent / "state"
    root.mkdir(parents=True, exist_ok=True)
    for t in TOPS: (root / t).mkdir(parents=True, exist_ok=True)
    (root / "Quarks" / "Blocks").mkdir(parents=True, exist_ok=True)
    cfg_path = root / "config.json"
    cfg = safe_load_json(cfg_path, default={}) or {}
    merged = DEFAULT_CONFIG.copy()
    merged.update(cfg)
    atomic_write_json(cfg_path, merged)
    return root

def load_node(path: Path) -> dict:
    return safe_load_json(path, default={})

def save_node(path: Path, obj: dict) -> None:
    obj.setdefault("meta", {})["updated_at"] = time.time()
    atomic_write_json(path, obj)

def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-\_]+", "-", name.strip().lower()).strip("-")
    return s or "unnamed"

def node_schema(cfg: dict, kind: str, human_name: str, size=(18,18)) -> dict:
    rows, cols = size
    symbol = cfg.get("symbols",{}).get("levels",{}).get(kind, "?")
    grid = [[{ "txt":"", "t": None, "ref": None, "meta": {} } for _ in range(cols)] for __ in range(rows)]
    return {"#": human_name, "id": slugify(human_name), "kind": kind, "symbol": symbol,
            "grid_size": [rows, cols], "grid": grid, "meta": {"created_at": time.time(), "updated_at": time.time()}, "links": []}

# ----------------------------- Cell helpers --------------------------------
def add_history(cell: dict, action: str, by: str = "local") -> None:
    meta = cell.setdefault("meta", {})
    hist = meta.setdefault("history", [])
    hist.append({"ts": int(time.time()), "action": action, "by": by})
    if len(hist) > HISTORY_CAP:
        del hist[0:len(hist)-HISTORY_CAP]

def is_marked(cell: dict) -> bool:
    return bool((cell.get("meta") or {}).get("mark"))

def set_mark(cell: dict, val: bool) -> None:
    cell.setdefault("meta", {})["mark"] = bool(val)
    add_history(cell, f"mark {'on' if val else 'off'}")

def cell_from_addr(addr: str) -> Tuple[int,int]:
    raw = addr
    s = str(addr).strip().upper()
    s = re.sub(r'[^A-Z0-9]', '', s)
    m = re.fullmatch(r'([A-Z]+)(\d+)', s)
    if not m:
        raise ValueError(f"Bad cell address '{raw}'. Try like B7.")
    col_str, row_str = m.groups()
    col = 0
    for ch in col_str:
        col = col * 26 + (ord(ch) - 64)
    col -= 1
    row = int(row_str) - 1
    return (row, col)

def cell_rc_valid(node: dict, r: int, c: int) -> bool:
    rows, cols = node.get("grid_size",[18,18])
    return 0 <= r < rows and 0 <= c < cols

# ------------------------- Blocks / Orientation ----------------------------
DIRS = ['N','E','S','W']
ARROWS = {'N':'^','E':'>','S':'v','W':'<'}

def rotate_dir(d: str, rot: int) -> str:
    idx = DIRS.index(d)
    steps = (rot // 90) % 4
    return DIRS[(idx + steps) % 4]

def get_block_instance(cell: dict) -> Optional[dict]:
    return (cell.get("meta") or {}).get("block")

def set_block_instance(cell: dict, block_id: str, rot: int = 0):
    b = {"id": block_id, "rot": int(rot) % 360}
    cell.setdefault("meta", {})["block"] = b
    add_history(cell, f"place block {block_id} rot {b['rot']}")

def clear_block_instance(cell: dict):
    if (cell.get("meta") or {}).get("block"):
        add_history(cell, "clear block")
    (cell.get("meta") or {}).pop("block", None)

def load_block_template(root: Path, block_id: str) -> Optional[dict]:
    p = root / "Quarks" / "Blocks" / f"{block_id}.json"
    if p.exists():
        return load_node(p)
    return None

def block_ports(block_tpl: dict) -> Dict[str, List[Tuple[int,int,str]]]:
    rows, cols = block_tpl.get("grid_size", [3,3])
    ports_in, ports_out = [], []
    for r in range(rows):
        for c in range(cols):
            txt = (block_tpl["grid"][r][c].get("txt") or "").strip()
            if txt == "?":
                d = nearest_edge_dir(r, c, rows, cols)
                ports_in.append((r,c,d))
            elif txt == "~":
                d = nearest_edge_dir(r, c, rows, cols)
                ports_out.append((r,c,d))
    return {'in': ports_in, 'out': ports_out}

def nearest_edge_dir(r, c, rows, cols) -> str:
    top = r
    left = c
    bottom = rows - 1 - r
    right = cols - 1 - c
    m = min(top, right, bottom, left)
    if m == top: return 'N'
    if m == right: return 'E'
    if m == bottom: return 'S'
    return 'W'

# --------------------------------------------------------------------------
# Helpers to resolve the effective type ("t") through block templates and
# reference chains. Quark- and Atom-level cells can derive their displayed
# type from their own 't', from a block's B2 cell, or via a reference to
# another node's cell. These helpers search across all top-level namespaces.

def find_node_by_id(app_root: Path, node_id: str) -> Optional[dict]:
    """Locate and load a node JSON by its slug id across all levels and namespaces."""
    # Search for the file named {node_id}.json anywhere under each top-level folder
    for level in TOPS:
        base = app_root / level
        if not base.exists():
            continue
        # Search recursively; stop at first match
        for p in base.rglob(f"{node_id}.json"):
            try:
                return load_node(p)
            except Exception:
                continue
    return None

def get_cell_by_addr(node: dict, addr: str) -> Optional[dict]:
    """Return the cell dict at a given address within a node, if valid."""
    try:
        r, c = cell_from_addr(addr)
        if cell_rc_valid(node, r, c):
            return node.get("grid", [])[r][c]
    except Exception:
        pass
    return None

def get_inherited_t(node: dict, cell: dict, app_root: Path, visited: Optional[set] = None) -> Optional[str]:
    """Resolve the effective type ('t') for a cell.

    Checks the cell's own 't' value, then any block instance's B2 cell,
    then recursively follows a reference to another node's cell. A set of
    visited node ids prevents infinite recursion due to cycles."""
    if visited is None:
        visited = set()
    # Direct value
    t_val = cell.get("t")
    if t_val:
        return t_val
    # If there is a block instance, consult the block template's B2 cell (row 1, col 1)
    blk = get_block_instance(cell)
    if blk:
        tpl = load_block_template(app_root, blk.get("id", ""))
        if tpl:
            rows, cols = tpl.get("grid_size", [3,3])
            if rows > 1 and cols > 1:
                block_cell = tpl.get("grid", [])[1][1]
                t_b = get_inherited_t(tpl, block_cell, app_root, visited)
                if t_b:
                    return t_b
    # Follow a reference if present
    ref = cell.get("ref")
    if ref:
        ref_str = str(ref).strip()
        # References are strings like '@nodeId[#CellAddr]'. If no cell address is provided,
        # default to B2. The '#' character may be omitted.
        if ref_str.startswith("@"):  # only process refs beginning with '@'
            body = ref_str[1:]
            nid = None
            addr = None
            if "[" in body and body.endswith("]"):
                nid, addr = body.split("[", 1)
                addr = addr.rstrip("]").strip()
                if not addr:
                    addr = "B2"
            else:
                nid = body.strip()
                addr = "B2"
            if nid and nid not in visited:
                visited.add(nid)
                ref_node = find_node_by_id(app_root, nid)
                if ref_node:
                    # Ensure address is present; default to B2 if not specified
                    cell_addr = addr or "B2"
                    ref_cell = get_cell_by_addr(ref_node, cell_addr)
                    if ref_cell:
                        t_r = get_inherited_t(ref_node, ref_cell, app_root, visited)
                        if t_r:
                            return t_r
    return None

def cell_arrow_for_block(block: dict, tpl: dict) -> str:
    rot = int(block.get("rot",0)) % 360
    pr = block_ports(tpl)
    out_dirs = [d for _,_,d in pr['out']]
    if out_dirs:
        facing = rotate_dir(out_dirs[0], rot)
    else:
        facing = rotate_dir('N', rot)
    return ARROWS.get(facing, '^')

def diagonal_conflicts(node: dict, r:int, c:int) -> List[Tuple[int,int]]:
    res = []
    for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        rr, cc = r+dr, c+dc
        if cell_rc_valid(node, rr, cc):
            nb = (node["grid"][rr][cc].get("meta") or {}).get("block")
            if nb:
                res.append((rr,cc))
    return res

# ------------------------------ Rendering ----------------------------------
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def has_history(cell: dict) -> bool:
    hist = (cell.get("meta") or {}).get("history") or []
    return len(hist) > 0

def cell_display(node: dict, cell: dict, app_root: Path) -> str:
    kind = node.get("kind","quark")
    cfg = safe_load_json(app_root / "config.json", default=DEFAULT_CONFIG)
    display_mode = cfg.get("ui",{}).get("display_mode","normal")
    # Get the effective type for this cell, possibly inherited via block or reference
    t_eff = get_inherited_t(node, cell, app_root) or ""
    txt = (cell.get("txt") or "").strip()
    ref = (cell.get("ref") or "").strip()
    marked = is_marked(cell)
    block = get_block_instance(cell) if kind == "synthesis" else None
    inner = " "
    if kind == "quark":
        # For quark cells, show the first letter of the type if present; otherwise fallback
        if t_eff:
            inner = str(t_eff)[0].upper()
        else:
            inner = "." if (marked or txt or ref) else " "
    elif kind == "atom":
        # For atom cells, show the first letter of the type if present; otherwise fallback
        if t_eff:
            inner = str(t_eff)[0].upper()
        else:
            inner = "^" if (marked or txt or ref) else " "
    else:
        if display_mode == "arrows" and block:
            tpl = load_block_template(app_root, block.get("id",""))
            inner = cell_arrow_for_block(block, tpl) if tpl else "?"
        else:
            if t_eff:
                inner = str(t_eff)[0].upper()
            elif txt:
                inner = txt[:1].upper()
            elif ref:
                inner = "@"
            elif block and marked:
                inner = "."
            elif marked:
                inner = "."
            else:
                inner = " "
    br_l, br_r = ("{", "}") if has_history(cell) else ("[", "]")
    return f"{br_l}{inner}{br_r}"

def draw_grid(node: dict, app_root: Path) -> None:
    rows, cols = node.get("grid_size",[18,18])
    colw = 4
    hdr = "    " + "".join(f"{(chr(65 + (i%26))).ljust(colw)}" for i in range(cols))
    print(hdr)
    for r in range(rows):
        line = f"{r+1:2d} "
        for c in range(cols):
            cell = node["grid"][r][c]
            token = cell_display(node, cell, app_root)
            line += token.ljust(colw)
        print(line)

# ------------------------------- UI (CLI) ---------------------------------
def prompt(msg: str) -> str:
    return input(msg).strip()

def ensure_namespace(root: Path, level: str, ns: str) -> Path:
    if level == "Blocks (under Quarks)":
        base = root / "Quarks" / "Blocks" / (ns if ns != "default" else "")
    else:
        base = root / level / (ns if ns != "default" else "")
    base.mkdir(parents=True, exist_ok=True)
    return base

def list_nodes(root: Path, level: str, ns: str) -> List[Path]:
    base = ensure_namespace(root, level, ns)
    return sorted(base.glob("*.json"))

def view_cell(node: dict, addr: str) -> None:
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print(f"Cell out of range for this grid: {addr.upper()}"); return
    cell = node["grid"][r][c]
    print(json.dumps({"addr": addr.upper(), **cell}, indent=2))

def view_all(node: dict) -> None:
    rows, cols = node.get("grid_size",[18,18])
    out = []
    for r in range(rows):
        for c in range(cols):
            cell = node["grid"][r][c]
            if is_marked(cell) or cell.get("t") or cell.get("ref") or (cell.get("txt") or "").strip() or get_block_instance(cell):
                addr = f"{chr(65+(c%26))}{r+1}"
                out.append({"addr": addr, **cell})
    if not out:
        print("(all cells empty)")
    else:
        print(json.dumps(out, indent=2))

def history_cell(node: dict, addr: str) -> None:
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print(f"Cell out of range for this grid: {addr.upper()}"); return
    cell = node["grid"][r][c]
    hist = (cell.get("meta") or {}).get("history") or []
    if not hist:
        print("No history for this cell."); return
    for ev in reversed(hist):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.get("ts",0)))
        print(f"[{ts}] {ev.get('action','')} (by: {ev.get('by','?')})")

# -------------------------- Commands: editing -------------------------------
def cmd_mark(node: dict, addr: str, val: str):
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print("Cell out of range."); return
    v = val.lower() in ("on","1","true","yes","y")
    set_mark(node["grid"][r][c], v)

def cmd_set(node: dict, addr: str, field: str, rest: str):
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print("Cell out of range."); return
    cell = node["grid"][r][c]
    kind = node.get("kind","quark")
    if field == "txt":
        cell["txt"] = rest
        add_history(cell, f"set txt {rest[:24]}")
    elif field == "ref":
        val = rest.strip()
        if val == "-" or val == "":
            cell["ref"] = None; add_history(cell, "clear ref")
        else:
            cell["ref"] = val; add_history(cell, f"set ref {val[:24]}")
    elif field == "type":
        # Accept any text for the type field; '-' or empty clears it
        val = rest.strip()
        if val == "-" or val == "":
            cell["t"] = None
            add_history(cell, "clear type")
        else:
            cell["t"] = val
            add_history(cell, f"set type {val}")
    else:
        print("Unknown field.")

# ------------------------ Commands: blocks & wires --------------------------
def cmd_mode(root: Path, mode: str):
    cfg_path = root / "config.json"
    cfg = safe_load_json(cfg_path, default=DEFAULT_CONFIG)
    if mode not in ("normal","arrows"):
        print("Mode must be 'normal' or 'arrows'"); return
    cfg.setdefault("ui",{})["display_mode"] = mode
    atomic_write_json(cfg_path, cfg)
    print(f"Display mode set to: {mode}")

def cmd_place_block(node: dict, root: Path, addr: str, block_id: str, rot: int=0):
    if node.get("kind") != "synthesis":
        print("Blocks can only be placed in Synthesis."); return
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print("Cell out of range."); return
    tpl = load_block_template(root, block_id)
    if not tpl or tpl.get("kind") != "block":
        print("Block template not found or not a block."); return
    # corner-touch rule
    cfg = safe_load_json(root / "config.json", default=DEFAULT_CONFIG)
    rule = (cfg.get("blocks",{}) or {}).get("no_corner_touch", True)
    if rule:
        conflicts = diagonal_conflicts(node, r, c)
        if conflicts:
            print("Placement blocked: diagonal blocks present. Use 'auto-guard' or place a replicator.")
            return
    set_block_instance(node["grid"][r][c], block_id, rot)

def cmd_orient(node: dict, addr: str, how: str):
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print("Cell out of range."); return
    cell = node["grid"][r][c]
    b = get_block_instance(cell)
    if not b:
        print("No block at this cell."); return
    rot = int(b.get("rot",0)) % 360
    if how == "cw": rot = (rot + 90) % 360
    elif how == "ccw": rot = (rot + 270) % 360
    elif how == "180": rot = (rot + 180) % 360
    else:
        try: rot = (int(how) % 360)
        except: print("orient needs cw|ccw|180 or degrees"); return
    b["rot"] = rot
    add_history(cell, f"orient {rot}")

def cmd_front(node: dict, addr: str, d: str):
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print("Cell out of range."); return
    cell = node["grid"][r][c]
    b = get_block_instance(cell)
    if not b:
        print("No block at this cell."); return
    d = d.upper()
    if d not in ('N','E','S','W'):
        print("front needs N|E|S|W"); return
    cell.setdefault("meta", {})["front"] = d
    add_history(cell, f"front {d}")

def cmd_clear_block(node: dict, addr: str):
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print("Cell out of range."); return
    clear_block_instance(node["grid"][r][c])

def cmd_show_ports(node: dict, root: Path, addr: str):
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print("Cell out of range."); return
    cell = node["grid"][r][c]
    b = get_block_instance(cell)
    if not b:
        print("No block at this cell."); return
    tpl = load_block_template(root, b.get("id",""))
    if not tpl:
        print("Template not found."); return
    rot = int(b.get("rot",0)) % 360
    pr = block_ports(tpl)
    dirs_in = [rotate_dir(d, rot) for _,_,d in pr['in']]
    dirs_out = [rotate_dir(d, rot) for _,_,d in pr['out']]
    print({"in": dirs_in, "out": dirs_out})

def cmd_wire(node: dict, src: str, dst: str):
    # src like "B7.out", dst like "B8.inN" (port suffix optional)
    node.setdefault("links", [])
    node["links"].append({"from": src, "to": dst})
    print(f"Linked {src} -> {dst}")

def cmd_unwire(node: dict, src: str):
    links = node.get("links") or []
    node["links"] = [lk for lk in links if lk.get("from") != src]
    print(f"Unwired {src}")

def cmd_auto_guard(node: dict, root: Path, addr: str):
    # Place replicator diagonally around a cell to satisfy no_corner_touch
    (r,c) = cell_from_addr(addr)
    if not cell_rc_valid(node, r, c):
        print("Cell out of range."); return
    cfg = safe_load_json(root / "config.json", default=DEFAULT_CONFIG)
    repl_id = (cfg.get("blocks",{}) or {}).get("replicator_id","replicator")
    for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        rr, cc = r+dr, c+dc
        if cell_rc_valid(node, rr, cc):
            cell = node["grid"][rr][cc]
            if not get_block_instance(cell):
                set_block_instance(cell, repl_id, 0)
    print("Auto-guard placed (replicators in diagonals where empty).")

# --------------------------------- Menu ------------------------------------
def clear_and_draw(node: dict, path: Path, app_root: Path):
    clear_screen()
    print(f"{node.get('#')}  ({node.get('kind')})     File: {path}")
    draw_grid(node, app_root)
    print("""
Commands:
  mark <Cell> on|off
  set <Cell> txt <text>
  set <Cell> ref <@id[#Cell] | ->
  set <Cell> type <text|->
      Set the type of the cell to any text string (only first letter shown in grid),
      or '-' to clear the type.
  mode <normal|arrows>
  place <Cell> block <id> [rot <0|90|180|270>]
  front <Cell> <N|E|S|W>
  orient <Cell> <cw|ccw|180|0|90|180|270>
  clear <Cell> block
  show ports <Cell>
  wire <SRC> -> <DST>     # e.g., wire B7.out -> B8.inN
  unwire <SRC>
  auto-guard <Cell>
  view <Cell> | view all | history <Cell>
  compile
  save | back
""")

def edit_node(app_root: Path, cfg: dict, path: Path) -> None:
    node = load_node(path)
    if not node:
        print("Failed to load node."); return
    while True:
        clear_and_draw(node, path, app_root)
        cmd = prompt("> ")
        if not cmd:
            continue

        if cmd == "back":
            break

        if cmd == "save":
            save_node(path, node); print(f"Saved: {path}"); time.sleep(0.5); continue

        if cmd.startswith("view all"):
            clear_screen(); print(f"VIEW ALL — {node.get('#')}"); view_all(node); input("\n(enter to return)"); continue

        if cmd.startswith("view "):
            try:
                _, addr = cmd.split(" ",1)
                clear_screen(); print(f"VIEW — {addr.upper()}"); view_cell(node, addr); input("\n(enter to return)")
            except Exception as e:
                print(f"Error: {e}"); time.sleep(0.6)
            continue

        if cmd.startswith("history "):
            try:
                _, addr = cmd.split(" ",1)
                clear_screen(); print(f"HISTORY — {addr.upper()}"); history_cell(node, addr); input("\n(enter to return)")
            except Exception as e:
                print(f"Error: {e}"); time.sleep(0.6)
            continue

        if cmd.startswith("mark "):
            try:
                _, a, v = cmd.split(" ",2)
                cmd_mark(node, a, v)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("set "):
            try:
                _, a, field, rest = cmd.split(" ",3)
                cmd_set(node, a, field, rest)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("mode "):
            try:
                _, m = cmd.split(" ",1)
                cmd_mode(app_root, m)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("place "):
            # place <Cell> block <id> [rot x]
            try:
                parts = cmd.split()
                if len(parts) >= 4 and parts[2] == "block":
                    addr = parts[1]; bid = parts[3]; rot = 0
                    if len(parts) >= 6 and parts[4] == "rot":
                        rot = int(parts[5])
                    cmd_place_block(node, app_root, addr, bid, rot)
                else:
                    print("Usage: place <Cell> block <id> [rot <0|90|180|270>]")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("front "):
            try:
                _, a, d = cmd.split(" ",2)
                cmd_front(node, a, d)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("orient "):
            try:
                _, a, how = cmd.split(" ",2)
                cmd_orient(node, a, how)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("clear "):
            try:
                _, a, what = cmd.split(" ",2)
                if what == "block":
                    cmd_clear_block(node, a)
                else:
                    print("Usage: clear <Cell> block")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("show ports "):
            try:
                _, _, a = cmd.split(" ",2)
                cmd_show_ports(node, app_root, a)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("wire "):
            try:
                # wire SRC -> DST
                m = re.match(r"wire\s+([^\s]+)\s*->\s*([^\s]+)", cmd)
                if not m:
                    print("Usage: wire <SRC> -> <DST>");
                else:
                    src, dst = m.group(1), m.group(2)
                    cmd_wire(node, src, dst)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("unwire "):
            try:
                _, src = cmd.split(" ",1)
                cmd_unwire(node, src)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd.startswith("auto-guard "):
            try:
                _, a = cmd.split(" ",1)
                cmd_auto_guard(node, app_root, a)
            except Exception as e:
                print(f"Error: {e}")
            continue

        if cmd == "compile":
            prog = pyenexe.compile_synthesis(node)
            outp = path.with_suffix(".prog.json")
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(prog, f, indent=2)
            print(f"Compiled to: {outp}")
            time.sleep(0.6)
            continue

        print("Unknown command.")

# ----------------------- Browser / creation flow ---------------------------
def list_namespaces(root: Path, level: str) -> List[str]:
    base = root / ( "Quarks" if level == "Blocks (under Quarks)" else level )
    nss = set(["default"])
    if base.exists():
        for p in base.rglob("*.json"):
            rel = p.relative_to(base)
            if rel.parent != Path("."):
                nss.add(str(rel.parent).replace("\\", "/"))
        for d in base.rglob("*"):
            if d.is_dir() and d != base:
                nss.add(str(d.relative_to(base)).replace("\\", "/"))
    return sorted(nss)

def choose_level() -> str:
    print("Select level:")
    levels = TOPS + ["Blocks (under Quarks)"]
    for i, t in enumerate(levels, 1):
        print(f"  {i}) {t}")
    s = input("Select #: ").strip()
    try:
        i = int(s)
        if 1 <= i <= len(levels):
            return levels[i-1]
    except:
        pass
    return ""

def ensure_namespace(root: Path, level: str, ns: str) -> Path:
    if level == "Blocks (under Quarks)":
        base = root / "Quarks" / "Blocks" / (ns if ns != "default" else "")
    else:
        base = root / level / (ns if ns != "default" else "")
    base.mkdir(parents=True, exist_ok=True)
    return base

def create_node(root: Path, cfg: dict, level: str, ns: str) -> Optional[Path]:
    human = input('Enter "# name": ').strip()
    if not human:
        print("Name required.")
        return None
    if level == "Blocks (under Quarks)":
        kind, size = "block", (3,3)
    else:
        kind = {"Quarks":"quark", "Atoms":"atom", "Synthesis":"synthesis"}[level]
        size = tuple(cfg.get("grid",{}).get(k, v) for k,v in (("rows",18),("cols",18)))
    obj = node_schema(cfg, kind, human, size=size)
    base = ensure_namespace(root, level, ns)
    path = base / f"{obj['id']}.json"
    if path.exists():
        print("File exists; choose another name.")
        return None
    save_node(path, obj)
    print(f"Created: {path}")
    return path

def list_nodes_with_labels(root: Path, level: str, ns: str):
    base = ensure_namespace(root, level, ns)
    nodes = sorted(base.glob("*.json"))
    labels = [f"{p.stem}  —  {safe_load_json(p,{}).get('#','(no name)')}" for p in nodes]
    labels.insert(0, "[+] New")
    return nodes, labels

def choose_namespace(root: Path, level: str) -> str:
    base_level = "Quarks" if level == "Blocks (under Quarks)" else level
    namespaces = list_namespaces(root, level)
    print(f"{level} namespaces:")
    for i, ns in enumerate(namespaces, 1):
        print(f"  {i}) {ns}")
    print(f"  {len(namespaces)+1}) [+] New namespace")
    print(f"  {len(namespaces)+2}) [-] Delete namespace")
    s = input("Select #: ").strip()
    try:
        i = int(s)
        if 1 <= i <= len(namespaces):
            return namespaces[i-1]
        if i == len(namespaces)+1:
            ns = input("New namespace name: ").strip()
            return ns or "default"
        if i == len(namespaces)+2:
            del_ns = input("Namespace to delete: ").strip() or ""
            if del_ns in ("", "default"):
                print("Refusing to delete 'default' or empty."); time.sleep(0.6)
                return "default"
            base = ensure_namespace(root, base_level, del_ns)
            if not base.exists():
                print("Namespace not found."); time.sleep(0.6)
                return "default"
            ok = input(f"Delete '{del_ns}' and ALL contents? (y/N): ").strip().lower()
            if ok == "y":
                shutil.rmtree(base, ignore_errors=True)
                print("Deleted."); time.sleep(0.6)
            return "default"
    except:
        pass
    return "default"

def menu(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DRAM9_On (On_4) Blocks + Arrows")
    parser.add_argument("--state", dest="state_dir", help="Override user storage directory")
    args = parser.parse_args(argv)

    app_root = resolve_user_root(args.state_dir)
    cfg = safe_load_json(app_root / "config.json", default=DEFAULT_CONFIG)

    while True:
        clear_screen()
        print(f"DRAM9_On — v{cfg.get('version', VERSION)}  [root: {app_root}]")
        lvl = choose_level()
        if not lvl:
            print("Goodbye."); return 0

        ns = choose_namespace(app_root, lvl)
        print(f"\n{lvl} / {ns}")
        nodes, labels = list_nodes_with_labels(app_root, lvl, ns)
        for i, lab in enumerate(labels, 1):
            print(f"  {i}) {lab}")
        s = input("Select #: ").strip()
        try:
            i = int(s)
        except:
            continue
        if i == 1:
            p = create_node(app_root, cfg, lvl, ns)
            if p: edit_node(app_root, cfg, p)
            continue
        idx = i - 2
        if 0 <= idx < len(nodes):
            edit_node(app_root, cfg, nodes[idx])

if __name__ == "__main__":
    try:
        raise SystemExit(menu())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
