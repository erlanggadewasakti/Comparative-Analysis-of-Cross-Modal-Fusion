"""Build ADEF v3.1 notebook from v3 by replacing/inserting cells."""
import nbformat as nbf
import os
import sys

SRC = "adef_co_attention_v3.ipynb"
DST = "adef_co_attention_v31.ipynb"
CELLS_DIR = r"C:\Users\ERLANG~1\AppData\Local\Temp\opencode\v31_cells"

def read_cell(filename):
    """Read a cell source .py file from the v31_cells directory."""
    path = os.path.join(CELLS_DIR, filename)
    with open(path, encoding='utf-8') as f:
        return f.read()

print("Reading v3 notebook...")
nb = nbf.read(SRC, as_version=4)
cells = list(nb.cells)

# Update title to v3.1
cells[0].source = cells[0].source.replace("ADEF v3", "ADEF v3.1")

# --- Replace cells 2, 12, 13, 14 ---
cells[2].source = read_cell("cell_2_cfg.py")
print("  Cell 2: CFG v3.1 replaced")

cells[12].source = read_cell("cell_12_train.py")
print("  Cell 12: train_one_seed v3.1 replaced (+checkpoint saving)")

cells[13].source = read_cell("cell_13_eval.py")
print("  Cell 13: evaluate_test v3.1 replaced (+probs collection)")

# --- Insert new cells after cell 13 (tune + ensemble) ---
# After inserting at position 14, old cells 14+ shift right
cell_13b = nbf.v4.new_code_cell()
cell_13b.source = read_cell("cell_13b_tune.py")
cells.insert(14, cell_13b)
print("  Cell 14: tune_class_scaling() inserted")

cell_13c = nbf.v4.new_code_cell()
cell_13c.source = read_cell("cell_13c_ensemble.py")
cells.insert(15, cell_13c)
print("  Cell 15: ensemble_evaluate() inserted")

# Now old cell 14 (E1) is at index 16. Replace it.
cells[16].source = read_cell("cell_14_exp.py")
print("  Cell 16: E1 v3.1 experiment replaced (checkpoints + tuning + ensemble)")

# Old cell 15 (ablations) is now at index 17 -- keep it but disable checkpoint overwrite
cells[17].source = cells[17].source.replace(
    "_, best_state, best_score, hist = train_one_seed(seed, overrides)",
    "_, best_state, best_score, hist = train_one_seed(seed, overrides, save_checkpoint=False)"
)
print("  Cell 17: ablations kept (save_checkpoint=False added)")

# --- Append new cell: E9 FILTER_CONFLICT_PAIRS=False ---
cell_18 = nbf.v4.new_code_cell()
cell_18.source = read_cell("cell_18_e9.py")
cells.append(cell_18)
print("  Cell 18: E9 FILTER_CONFLICT_PAIRS=False appended")

# --- Write notebook ---
nb.cells = cells
nbf.write(nb, DST)
print(f"\nV3.1 notebook written to: {os.path.abspath(DST)}")
print(f"Total cells: {len(cells)}")
