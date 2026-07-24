"""Build ADEF v3.2 notebook from v3.1 by replacing/inserting cells."""
import nbformat as nbf
import os

SRC = "adef_co_attention_v31.ipynb"
DST = "adef_co_attention_v32.ipynb"
V32_DIR = r"C:\Users\ERLANG~1\AppData\Local\Temp\opencode\v32_cells"

def read_cell(filename):
    path = os.path.join(V32_DIR, filename)
    with open(path, encoding='utf-8') as f:
        return f.read()

print("Reading v3.1 notebook...")
nb = nbf.read(SRC, as_version=4)
cells = list(nb.cells)

# Update title to v3.2
cells[0].source = cells[0].source.replace("ADEF v3.1", "ADEF v3.2")

# --- Replace cells 2 and 12 ---
cells[2].source = read_cell("cell_2_cfg_v32.py")
print("  Cell 2: CFG v3.2 replaced (+NEUTRAL_OVERSAMPLE)")

cells[12].source = read_cell("cell_12_train_v32.py")
print("  Cell 12: train_one_seed v3.2 replaced (+train_loader +ckpt_tag)")

# --- Insert D1, D2, E10 after cell 16 (E1) ---
cell_d1 = nbf.v4.new_code_cell()
cell_d1.source = read_cell("cell_d1_diagnostic.py")
cells.insert(17, cell_d1)
print("  Cell 17: D1 diagnostic inserted (conflict cross-tab + u/margin/g)")

cell_d2 = nbf.v4.new_code_cell()
cell_d2.source = read_cell("cell_d2_rules.py")
cells.insert(18, cell_d2)
print("  Cell 18: D2 neutral rules inserted (R1-R4)")

cell_e10 = nbf.v4.new_code_cell()
cell_e10.source = read_cell("cell_e10_oversample.py")
cells.insert(19, cell_e10)
print("  Cell 19: E10 oversampling experiment inserted")

# --- Fix E9 (now at index 21: 20=ablations, 21=E9) ---
cells[21].source = cells[21].source.replace(
    "model_full, best_state_full, best_score_full, hist_full = train_one_seed(seed=42)",
    "model_full, best_state_full, best_score_full, hist_full = train_one_seed(seed=42, save_checkpoint=False)"
)
print("  Cell 21: E9 fixed (save_checkpoint=False)")

# --- Write notebook ---
nb.cells = cells
nbf.write(nb, DST)
print(f"\nV3.2 notebook written to: {os.path.abspath(DST)}")
print(f"Total cells: {len(cells)}")
