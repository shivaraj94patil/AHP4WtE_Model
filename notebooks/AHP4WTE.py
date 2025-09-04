#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =========================
# AHP HIERARCHY BUILDER
# =========================

# --- Imports (minimal, no heavy deps) ---
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterable
import numpy as np
import pandas as pd
import re

# --- Core data structure for AHP nodes ---
@dataclass
class AHPNode:
    """
    A generic AHP node that can represent: goal, criterion, sub-criterion, deciding factor, or alternative.
    """
    name: str
    kind: str  # 'goal' | 'criterion' | 'subcriterion' | 'factor' | 'alternative'
    parent: Optional["AHPNode"] = None
    children: List["AHPNode"] = field(default_factory=list)

    # Optional metadata (useful later when scoring/normalizing)
    direction: Optional[str] = None  # 'max' (benefit) | 'min' (cost) | None if not applicable
    unit: Optional[str] = None       # e.g., 'EUR/Ton', '%', 'kg eq CO2/ton'

    # Placeholders for AHP math (we'll fill these later)
    pairwise: Optional[np.ndarray] = None
    local_weights: Optional[np.ndarray] = None

    # --- tree helpers ---
    def add(self, child: "AHPNode") -> "AHPNode":
        child.parent = self
        self.children.append(child)
        return child

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def path(self) -> List[str]:
        n, parts = self, []
        while n is not None:
            parts.append(n.name)
            n = n.parent
        return list(reversed(parts))

# --- Pretty printer for the hierarchy ---
def print_tree(node: AHPNode, indent: int = 0) -> None:
    bullet = {
        "goal": "🎯",
        "criterion": "◼︎",
        "subcriterion": "▪︎",
        "factor": "•",
        "alternative": "–",
    }.get(node.kind, "•")
    meta = []
    if node.direction: meta.append(f"dir={node.direction}")
    if node.unit: meta.append(f"unit={node.unit}")
    metas = f" [{' | '.join(meta)}]" if meta else ""
    print("  " * indent + f"{bullet} {node.name}{metas}")
    for c in node.children:
        print_tree(c, indent + 1)

# --- Registry helpers (find nodes, export factor table) ---
def iter_nodes(root: AHPNode) -> Iterable[AHPNode]:
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))

def factors_table(root: AHPNode) -> pd.DataFrame:
    rows = []
    for n in iter_nodes(root):
        if n.kind == "factor":
            rows.append({
                "id": slug("/".join(n.path())),  # stable id for pairwise inputs later
                "name": n.name,
                "path": " > ".join(n.path()),
                "direction": n.direction,
                "unit": n.unit,
            })
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    return df

def slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

# --- Build your specific hierarchy ---
def build_mswt_ahp_hierarchy() -> Dict[str, object]:
    # Goal
    root = AHPNode("Rank MSW-to-Energy Technology Alternatives", kind="goal")

    # Alternatives (11)
    alternatives = [
        "INC-E", "INC-H", "AD", "PYR", "GAS", "HTC", "Composting",
        "INC-E-CCS 85%", "INC-E-CCS 95%", "INC-H-CCS- 85%", "INC-H-CCS- 95%",
    ]
    alternative_nodes = [AHPNode(a, kind="alternative") for a in alternatives]

    # === Main Criteria ===
    tech = root.add(AHPNode("Technological", kind="criterion"))
    eco  = root.add(AHPNode("Economical", kind="criterion"))
    env  = root.add(AHPNode("Environmental", kind="criterion"))
    soc  = root.add(AHPNode("Socio-cultural", kind="criterion"))

    # --- Technological ---
    # Feedstock -> deciding factors
    feedstock = tech.add(AHPNode("Feedstock", kind="subcriterion"))
    feedstock.add(AHPNode("Moisture Content", kind="factor", direction="min", unit="%"))
    feedstock.add(AHPNode("Calorific Value", kind="factor", direction="max", unit="MJ/kg"))
    feedstock.add(AHPNode("Pre-Processing", kind="factor", direction="min"))  # qualitative/categorical → later mapping

    # Energy Output, Efficiencies as subcriteria with direct factors (one-to-one)
    energy_output = tech.add(AHPNode("Energy Output", kind="subcriterion"))
    energy_output.add(AHPNode("Energy Output (as factor)", kind="factor", direction="max"))

    efficiencies = tech.add(AHPNode("Efficiencies", kind="subcriterion"))
    efficiencies.add(AHPNode("Efficiencies (as factor)", kind="factor", direction="max"))

    # Scalability -> deciding factors
    scal = tech.add(AHPNode("Scalability", kind="subcriterion"))
    scal.add(AHPNode("Technology Complexity", kind="factor", direction="min"))
    scal.add(AHPNode("No. of Processing Steps", kind="factor", direction="min"))
    scal.add(AHPNode("Retrofitting Feasibility", kind="factor", direction="max"))
    scal.add(AHPNode("Technology Maturity", kind="factor", direction="max"))
    scal.add(AHPNode("Technological Readiness", kind="factor", direction="max"))
    scal.add(AHPNode("Commercial-Scale Plants in Country", kind="factor", direction="max"))

    # --- Economical ---
    eco.add(AHPNode("CAPEX", kind="factor", direction="min", unit="EUR/Ton"))
    eco.add(AHPNode("OPEX", kind="factor", direction="min", unit="EUR/Ton"))
    eco.add(AHPNode("LCOE",  kind="factor", direction="min", unit="EUR/MJ"))

    # --- Environmental ---
    env.add(AHPNode("Waste Volume Reduction Potential", kind="factor", direction="max", unit="%"))

    air = env.add(AHPNode("Air Pollutants", kind="subcriterion"))
    air.add(AHPNode("CO2", kind="factor", direction="min", unit="kg eq CO2/ton"))
    air.add(AHPNode("NOx", kind="factor", direction="min", unit="kg eq CO2/ton"))
    air.add(AHPNode("PM",  kind="factor", direction="min"))
    air.add(AHPNode("SO2", kind="factor", direction="min", unit="g/t"))

    byp = env.add(AHPNode("Byproduct", kind="subcriterion"))
    byp.add(AHPNode("Ash/Char Residues", kind="factor", direction="min", unit="kg/ton MSW"))
    byp.add(AHPNode("Trace Metals", kind="factor", direction="min", unit="mg/ton"))

    water = env.add(AHPNode("Water Pollutants", kind="subcriterion"))
    water.add(AHPNode("Organic Compounds", kind="factor", direction="min"))
    water.add(AHPNode("Heavy Metals", kind="factor", direction="min"))
    water.add(AHPNode("Nutrients (N,P)", kind="factor", direction="min"))
    water.add(AHPNode("Suspended Solids", kind="factor", direction="min"))

    # --- Socio-cultural ---
    soc.add(AHPNode("Employment / Automation Focused", kind="factor"))  # direction depends on policy goals; set later
    soc.add(AHPNode("Worker Exposure Risk", kind="factor", direction="min"))
    soc.add(AHPNode("NIMBY Effect", kind="factor", direction="min"))

    # Package everything and return
    return {
        "root": root,
        "alternatives": alternative_nodes,  # kept separate; later each factor will hold pairwise comps over these
    }

# ---- Build & preview the structure ----
ahp = build_mswt_ahp_hierarchy()
root = ahp["root"]
alts = ahp["alternatives"]

print("AHP Hierarchy (structure only):\n")
print_tree(root)

print("\nAlternatives (11):")
for a in alts:
    print(f"  - {a.name}")

# Table of deciding factors (handy for pairwise-data templates)
df_factors = factors_table(root)
print("\nFactor registry (first 10 rows):")
display(df_factors.head(10))

# If you want the full table for export later:
# df_factors.to_csv("ahp_deciding_factors_catalog.csv", index=False)




# In[2]:


# ==========================================
# AHP: solver + CAPEX example (11 alternatives)
# ==========================================

# ---- Imports ----
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterable, Tuple
import numpy as np
import pandas as pd
import re

# ---- AHP node (same as before) ----
@dataclass
class AHPNode:
    name: str
    kind: str  # 'goal' | 'criterion' | 'subcriterion' | 'factor' | 'alternative'
    parent: Optional["AHPNode"] = None
    children: List["AHPNode"] = field(default_factory=list)
    direction: Optional[str] = None  # 'max' | 'min' | None
    unit: Optional[str] = None
    pairwise: Optional[np.ndarray] = None       # pairwise comparison matrix (local)
    local_weights: Optional[np.ndarray] = None  # eigenvector weights (local, normalized)

    def add(self, child: "AHPNode") -> "AHPNode":
        child.parent = self
        self.children.append(child)
        return child

# ---- Hierarchy builder (same as before) ----
def build_mswt_ahp_hierarchy() -> Dict[str, object]:
    root = AHPNode("Rank MSW-to-Energy Technology Alternatives", kind="goal")

    # Alternatives (canonical names used across the project)
    alternatives = [
        "INC-E", "INC-H", "AD", "PYR", "GAS", "HTC", "Composting",
        "INC-E-CCS 85%", "INC-E-CCS 95%", "INC-H-CCS- 85%", "INC-H-CCS- 95%",
    ]
    alternative_nodes = [AHPNode(a, kind="alternative") for a in alternatives]

    # --- Main Criteria ---
    tech = root.add(AHPNode("Technological", kind="criterion"))
    eco  = root.add(AHPNode("Economical", kind="criterion"))
    env  = root.add(AHPNode("Environmental", kind="criterion"))
    soc  = root.add(AHPNode("Socio-cultural", kind="criterion"))

    # --- Technological ---
    feedstock = tech.add(AHPNode("Feedstock", kind="subcriterion"))
    feedstock.add(AHPNode("Moisture Content", kind="factor", direction="min", unit="%"))
    feedstock.add(AHPNode("Calorific Value", kind="factor", direction="max", unit="MJ/kg"))
    feedstock.add(AHPNode("Pre-Processing", kind="factor", direction="min"))

    energy_output = tech.add(AHPNode("Energy Output", kind="subcriterion"))
    energy_output.add(AHPNode("Energy Output (as factor)", kind="factor", direction="max"))

    efficiencies = tech.add(AHPNode("Efficiencies", kind="subcriterion"))
    efficiencies.add(AHPNode("Efficiencies (as factor)", kind="factor", direction="max"))

    scal = tech.add(AHPNode("Scalability", kind="subcriterion"))
    scal.add(AHPNode("Technology Complexity", kind="factor", direction="min"))
    scal.add(AHPNode("No. of Processing Steps", kind="factor", direction="min"))
    scal.add(AHPNode("Retrofitting Feasibility", kind="factor", direction="max"))
    scal.add(AHPNode("Technology Maturity", kind="factor", direction="max"))
    scal.add(AHPNode("Technological Readiness", kind="factor", direction="max"))
    scal.add(AHPNode("Commercial-Scale Plants in Country", kind="factor", direction="max"))

    # --- Economical ---  (we'll use CAPEX here)
    capex = eco.add(AHPNode("CAPEX", kind="factor", direction="min", unit="EUR/Ton"))
    eco.add(AHPNode("OPEX", kind="factor", direction="min", unit="EUR/Ton"))
    eco.add(AHPNode("LCOE", kind="factor", direction="min", unit="EUR/MJ"))

    # --- Environmental ---
    env.add(AHPNode("Waste Volume Reduction Potential", kind="factor", direction="max", unit="%"))
    air = env.add(AHPNode("Air Pollutants", kind="subcriterion"))
    air.add(AHPNode("CO2", kind="factor", direction="min", unit="kg eq CO2/ton"))
    air.add(AHPNode("NOx", kind="factor", direction="min", unit="kg eq CO2/ton"))
    air.add(AHPNode("PM",  kind="factor", direction="min"))
    air.add(AHPNode("SO2", kind="factor", direction="min", unit="g/t"))
    byp = env.add(AHPNode("Byproduct", kind="subcriterion"))
    byp.add(AHPNode("Ash/Char Residues", kind="factor", direction="min", unit="kg/ton MSW"))
    byp.add(AHPNode("Trace Metals", kind="factor", direction="min", unit="mg/ton"))
    water = env.add(AHPNode("Water Pollutants", kind="subcriterion"))
    water.add(AHPNode("Organic Compounds", kind="factor", direction="min"))
    water.add(AHPNode("Heavy Metals", kind="factor", direction="min"))
    water.add(AHPNode("Nutrients (N,P)", kind="factor", direction="min"))
    water.add(AHPNode("Suspended Solids", kind="factor", direction="min"))

    # --- Socio-cultural ---
    soc.add(AHPNode("Employment / Automation Focused", kind="factor"))
    soc.add(AHPNode("Worker Exposure Risk", kind="factor", direction="min"))
    soc.add(AHPNode("NIMBY Effect", kind="factor", direction="min"))

    return {"root": root, "alternatives": alternative_nodes, "capex_node": capex}

# ---- Pretty tree (optional) ----
def print_tree(node: AHPNode, indent: int = 0) -> None:
    icon = {"goal":"🎯","criterion":"◼︎","subcriterion":"▪︎","factor":"•","alternative":"–"}.get(node.kind,"•")
    meta = []
    if node.direction: meta.append(f"dir={node.direction}")
    if node.unit: meta.append(f"unit={node.unit}")
    print("  " * indent + f"{icon} {node.name}" + (f" [{' | '.join(meta)}]" if meta else ""))
    for c in node.children:
        print_tree(c, indent + 1)

# =====================================================
# AHP Solver (Saaty): eigenvector + consistency checks
# =====================================================

# Saaty Random Index (RI) for n=1..15
SAATY_RI = {
    1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41,
    9:1.45, 10:1.49, 11:1.51, 12:1.48, 13:1.56, 14:1.57, 15:1.59
}

def ahp_eigenvector(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """Principal right eigenvector (normalized to sum=1) and lambda_max."""
    # numerical guard: ensure matrix is positive reciprocal
    assert A.shape[0] == A.shape[1], "Pairwise matrix must be square"
    # eigen decomposition
    vals, vecs = np.linalg.eig(A)
    idx = np.argmax(vals.real)
    w = vecs[:, idx].real
    w = np.abs(w)  # avoid sign ambiguity
    w = w / w.sum()
    lambda_max = vals[idx].real
    return w, float(lambda_max)

def ahp_consistency(A: np.ndarray) -> Dict[str, float]:
    """Return CI, RI, CR and lambda_max."""
    n = A.shape[0]
    w, lam = ahp_eigenvector(A)
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = SAATY_RI.get(n, SAATY_RI[15])  # fall back to 1.59
    CR = CI / RI if RI > 0 else 0.0
    return {"lambda_max": lam, "CI": CI, "RI": RI, "CR": CR}

# ==============================================
# Data handling + Pairwise construction for CAPEX
# ==============================================

# Mapping from your spreadsheet labels → canonical alternative names
ALT_ALIASES = {
    "INC-Heating":"INC-H",
    "INC-Electricity":"INC-E",
    "INC-H+CCS 85%":"INC-H-CCS- 85%",
    "INC-H+CCS 95%":"INC-H-CCS- 95%",
    "INC-E+CCS 85%":"INC-E-CCS 85%",
    "INC-E+CCS 95%":"INC-E-CCS 95%",
}

def canon_name(s: str) -> str:
    s = s.strip()
    return ALT_ALIASES.get(s, s)

def parse_capex_table() -> Dict[str, float]:
    """
    Hard-code the CAPEX avg values you provided.
    If you prefer CSV/Excel later, replace this with a loader.
    """
    rows = [
        ("INC-Heating",      "348-508", 428),
        ("INC-Electricity",  "522-660", 591),
        ("AD",               "218-272", 245),
        ("PYR",              "476-508", 492),
        ("GAS",              "544-665", 604.5),
        ("HTC",              "275-353", 314),
        ("Composting",       "98-180",  139),
        ("INC-H+CCS 85%",    "350-510", 430),
        ("INC-H+CCS 95%",    "352-515", 434),
        ("INC-E+CCS 85%",    "527-666", 596.5),
        ("INC-E+CCS 95%",    "528-667", 597.5),
    ]
    avg = {canon_name(name): float(avg_val) for (name, rng, avg_val) in rows}
    return avg

def pairwise_from_cardinal_cost(values: Dict[str, float], order: List[str]) -> np.ndarray:
    """
    Construct a positive reciprocal pairwise matrix from cardinal 'cost' values.
    Preference (i over j) = cost_j / cost_i  (lower cost = better).
    """
    n = len(order)
    A = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            vij = values[order[j]] / values[order[i]]
            A[i, j] = vij
            A[j, i] = 1.0 / vij
    return A

# ===================
# Aggregation engine
# ===================
def rank_alternatives_from_factor(node_factor: AHPNode, alternatives: List[AHPNode], local_alt_weights: np.ndarray) -> pd.DataFrame:
    """
    Build a simple ranking table for one factor. (Multi-factor aggregation will follow same pattern.)
    """
    node_factor.local_weights = local_alt_weights  # store in the node
    names = [a.name for a in alternatives]
    df = pd.DataFrame({"Alternative": names, f"Weight @ {node_factor.name}": local_alt_weights})
    df["Rank"] = df[f"Weight @ {node_factor.name}"].rank(ascending=False, method="dense").astype(int)
    df = df.sort_values(by=f"Weight @ {node_factor.name}", ascending=False).reset_index(drop=True)
    return df

# ==========================
# ---- Run the CAPEX demo ---
# ==========================
built = build_mswt_ahp_hierarchy()
root = built["root"]
alts_nodes = built["alternatives"]
capex_node = built["capex_node"]

# 1) Get CAPEX averages
capex_avg = parse_capex_table()

# 2) Ensure all 11 alternatives are present
ALT_ORDER = [a.name for a in alts_nodes]  # canonical order from hierarchy
missing = [a for a in ALT_ORDER if a not in capex_avg]
if missing:
    raise ValueError(f"Missing CAPEX values for: {missing}")

# 3) Build pairwise matrix (cost → lower better)
A_capex = pairwise_from_cardinal_cost(capex_avg, ALT_ORDER)

# 4) Solve with Saaty's eigenvector method + consistency
w_capex, lam_capex = ahp_eigenvector(A_capex)
cons_capex = ahp_consistency(A_capex)

# 5) Ranking table
ranking_capex = rank_alternatives_from_factor(capex_node, alts_nodes, w_capex)

# 6) Display results
print("=== CAPEX pairwise consistency ===")
print(f"lambda_max = {cons_capex['lambda_max']:.6f}")
print(f"CI = {cons_capex['CI']:.6f}, RI = {cons_capex['RI']:.2f}, CR = {cons_capex['CR']:.6f}")
print("(For a ratio-perfect matrix built from cardinal data, CR = 0 by construction.)\n")

print("=== Ranking on CAPEX (lower cost → higher weight) ===")
display(ranking_capex)



# In[3]:


# ==========================================
# AHP with CAPEX + OPEX (Economical)
# ==========================================

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Iterable, Tuple
import numpy as np
import pandas as pd

# ---------- AHP core ----------
@dataclass
class AHPNode:
    name: str
    kind: str  # 'goal'|'criterion'|'subcriterion'|'factor'|'alternative'
    parent: Optional["AHPNode"] = None
    children: List["AHPNode"] = field(default_factory=list)
    direction: Optional[str] = None  # 'max'|'min'|None
    unit: Optional[str] = None
    pairwise: Optional[np.ndarray] = None
    local_weights: Optional[np.ndarray] = None

    def add(self, child: "AHPNode") -> "AHPNode":
        child.parent = self
        self.children.append(child)
        return child

def build_mswt_ahp_hierarchy() -> Dict[str, object]:
    root = AHPNode("Rank MSW-to-Energy Technology Alternatives", kind="goal")

    # Alternatives (canonical)
    alternatives = [
        "INC-E", "INC-H", "AD", "PYR", "GAS", "HTC", "Composting",
        "INC-E-CCS 85%", "INC-E-CCS 95%", "INC-H-CCS- 85%", "INC-H-CCS- 95%",
    ]
    alternative_nodes = [AHPNode(a, kind="alternative") for a in alternatives]

    tech = root.add(AHPNode("Technological", kind="criterion"))
    eco  = root.add(AHPNode("Economical", kind="criterion"))
    env  = root.add(AHPNode("Environmental", kind="criterion"))
    soc  = root.add(AHPNode("Socio-cultural", kind="criterion"))

    # Economical factors (we'll use these now)
    capex_node = eco.add(AHPNode("CAPEX", kind="factor", direction="min", unit="EUR/Ton"))
    opex_node  = eco.add(AHPNode("OPEX",  kind="factor", direction="min", unit="EUR/Ton"))
    eco.add(AHPNode("LCOE", kind="factor", direction="min", unit="EUR/MJ"))  # placeholder (no data yet)

    return {"root": root, "alternatives": alternative_nodes, "eco": eco,
            "capex_node": capex_node, "opex_node": opex_node}

# ---------- Pretty tree (optional) ----------
def print_tree(node: AHPNode, indent: int = 0) -> None:
    icon = {"goal":"🎯","criterion":"◼︎","subcriterion":"▪︎","factor":"•","alternative":"–"}.get(node.kind,"•")
    meta = []
    if node.direction: meta.append(f"dir={node.direction}")
    if node.unit: meta.append(f"unit={node.unit}")
    print("  " * indent + f"{icon} {node.name}" + (f" [{' | '.join(meta)}]" if meta else ""))
    for c in node.children:
        print_tree(c, indent + 1)

# ---------- Saaty eigenvector + consistency ----------
SAATY_RI = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49,11:1.51,12:1.48,13:1.56,14:1.57,15:1.59}

def ahp_eigenvector(A: np.ndarray) -> Tuple[np.ndarray, float]:
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    return w, float(vals[k].real)

def ahp_consistency(A: np.ndarray) -> Dict[str, float]:
    n = A.shape[0]
    w, lam = ahp_eigenvector(A)
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = SAATY_RI.get(n, SAATY_RI[15])
    CR = CI / RI if RI > 0 else 0.0
    return {"lambda_max": lam, "CI": CI, "RI": RI, "CR": CR}

# ---------- Data (from your tables) ----------
ALT_ALIASES = {
    "INC-Heating":"INC-H",
    "INC-Electricity":"INC-E",
    "INC-H+CCS 85%":"INC-H-CCS- 85%",
    "INC-H+CCS 95%":"INC-H-CCS- 95%",
    "INC-E+CCS 85%":"INC-E-CCS 85%",
    "INC-E+CCS 95%":"INC-E-CCS 95%",
}
def canon(s: str) -> str: return ALT_ALIASES.get(s.strip(), s.strip())

def capex_avg_table() -> Dict[str, float]:
    rows = [
        ("INC-Heating",     428),
        ("INC-Electricity", 591),
        ("AD",              245),
        ("PYR",             492),
        ("GAS",             604.5),
        ("HTC",             314),
        ("Composting",      139),
        ("INC-H+CCS 85%",   430),
        ("INC-H+CCS 95%",   434),
        ("INC-E+CCS 85%",   596.5),
        ("INC-E+CCS 95%",   597.5),
    ]
    return {canon(n): float(v) for n, v in rows}

def opex_avg_table() -> Dict[str, float]:
    rows = [
        ("INC-Heating",     30),
        ("INC-Electricity", 33),
        ("AD",              38.5),
        ("PYR",             49.5),
        ("GAS",             61.5),
        ("HTC",             35.5),
        ("Composting",      12.5),
        ("INC-H+CCS 85%",   72),
        ("INC-H+CCS 95%",   73),
        ("INC-E+CCS 85%",   76),
        ("INC-E+CCS 95%",   77.5),
    ]
    return {canon(n): float(v) for n, v in rows}

# Build pairwise matrix from cardinal *cost* values: pref(i over j) = cost_j / cost_i
def pairwise_from_cost(values: Dict[str, float], order: List[str]) -> np.ndarray:
    n = len(order)
    A = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            r = values[order[j]] / values[order[i]]
            A[i, j] = r
            A[j, i] = 1.0 / r
    return A

# ---------- Aggregation helpers ----------
def ranking_df(names: List[str], weights: np.ndarray, label: str) -> pd.DataFrame:
    df = pd.DataFrame({"Alternative": names, label: weights})
    df["Rank"] = df[label].rank(ascending=False, method="dense").astype(int)
    return df.sort_values(by=label, ascending=False).reset_index(drop=True)

# ========================= Run =========================
built = build_mswt_ahp_hierarchy()
root, eco = built["root"], built["eco"]
alts = [a.name for a in built["alternatives"]]
capex_node, opex_node = built["capex_node"], built["opex_node"]

capex_avg = capex_avg_table()
opex_avg  = opex_avg_table()

# sanity: ensure data coverage
for a in alts:
    assert a in capex_avg, f"Missing CAPEX for {a}"
    assert a in opex_avg,  f"Missing OPEX for {a}"

# Pairwise matrices (ratio data => perfectly consistent)
A_capex = pairwise_from_cost(capex_avg, alts)
A_opex  = pairwise_from_cost(opex_avg,  alts)

w_capex, _ = ahp_eigenvector(A_capex)
w_opex,  _ = ahp_eigenvector(A_opex)

cons_capex = ahp_consistency(A_capex)
cons_opex  = ahp_consistency(A_opex)

print("=== Consistency ===")
print(f"CAPEX: lambda={cons_capex['lambda_max']:.6f}  CI={cons_capex['CI']:.6f}  CR={cons_capex['CR']:.6f}")
print(f"OPEX : lambda={cons_opex['lambda_max']:.6f}   CI={cons_opex['CI']:.6f}   CR={cons_opex['CR']:.6f}")
print("(Ratio matrices from cardinal data are consistency-perfect → CR=0.)\n")

# Factor weights within Economical (CAPEX vs OPEX). Use equal importance by default.
A_factors = np.array([[1.0, 1.0],
                      [1.0, 1.0]])
w_factors, _ = ahp_eigenvector(A_factors)  # -> [0.5, 0.5]

# Aggregate Economical score for each alternative
econ_weights = w_factors[0] * w_capex + w_factors[1] * w_opex

# ---------- Outputs ----------
capex_rank = ranking_df(alts, w_capex, "Weight @ CAPEX (lower better)")
opex_rank  = ranking_df(alts, w_opex,  "Weight @ OPEX (lower better)")
econ_rank  = ranking_df(alts, econ_weights, "Weight @ Economical (CAPEX+OPEX)")

print("=== Ranking: CAPEX (lower cost → higher weight) ===")
display(capex_rank)

print("\n=== Ranking: OPEX (lower cost → higher weight) ===")
display(opex_rank)

print("\n=== Ranking: ECONOMICAL (equal CAPEX & OPEX) ===")
display(econ_rank)

# If you want CSV exports:
# capex_rank.to_csv("ranking_capex.csv", index=False)
# opex_rank.to_csv("ranking_opex.csv",  index=False)
# econ_rank.to_csv("ranking_economical.csv", index=False)


# In[4]:


# =================================================
# AHP Economical (CAPEX + OPEX + LCOE with weights)
# =================================================

import numpy as np
import pandas as pd

# --- Saaty eigenvector method ---
def ahp_eigenvector(A):
    vals, vecs = np.linalg.eig(A)
    idx = np.argmax(vals.real)
    w = np.abs(vecs[:, idx].real)
    w = w / w.sum()
    return w, vals[idx].real

# --- Data (from your latest table) ---
alts = [
    "INC-H","INC-E","AD","PYR","GAS","HTC","Composting",
    "INC-H-CCS- 85%","INC-H-CCS- 95%","INC-E-CCS 85%","INC-E-CCS 95%"
]

capex_avg = {
    "INC-H":428,"INC-E":591,"AD":245,"PYR":492,"GAS":604.5,"HTC":314,"Composting":139,
    "INC-H-CCS- 85%":430,"INC-H-CCS- 95%":434,"INC-E-CCS 85%":596.5,"INC-E-CCS 95%":597.5
}
opex_avg = {
    "INC-H":30,"INC-E":33,"AD":38.5,"PYR":49.5,"GAS":61.5,"HTC":35.5,"Composting":12.5,
    "INC-H-CCS- 85%":72,"INC-H-CCS- 95%":73,"INC-E-CCS 85%":76,"INC-E-CCS 95%":77.5
}
lcoe_avg = {
    "INC-H":0.007615,"INC-E":0.036794,"AD":0.029,"PYR":0.020301,"GAS":0.016986,"HTC":0.008752,
    "Composting":np.nan,  # NA → excluded
    "INC-H-CCS- 85%":0.014841,"INC-H-CCS- 95%":0.016188,"INC-E-CCS 85%":0.06719,"INC-E-CCS 95%":0.095444
}

# --- Pairwise matrix from cost values ---
def pairwise_from_cost(values, order):
    n = len(order)
    A = np.ones((n,n))
    for i in range(n):
        for j in range(i+1,n):
            vi, vj = values[order[i]], values[order[j]]
            if np.isnan(vi) or np.isnan(vj):
                vij = 1  # treat missing equally
            else:
                vij = vj / vi
            A[i,j] = vij
            A[j,i] = 1/vij if vij!=0 else 1
    return A

# --- Build and solve ---
A_capex = pairwise_from_cost(capex_avg, alts)
A_opex  = pairwise_from_cost(opex_avg,  alts)
A_lcoe  = pairwise_from_cost(lcoe_avg,  alts)

w_capex,_ = ahp_eigenvector(A_capex)
w_opex,_  = ahp_eigenvector(A_opex)
w_lcoe,_  = ahp_eigenvector(A_lcoe)

# Factor-level comparison (CAPEX vs OPEX vs LCOE)
A_factors = np.array([
    [1, 1, 1/3],
    [1, 1, 1/3],
    [3, 3, 1]
], dtype=float)
w_factors,_ = ahp_eigenvector(A_factors)

print("Factor weights (Economical):")
print(f"CAPEX={w_factors[0]:.3f}, OPEX={w_factors[1]:.3f}, LCOE={w_factors[2]:.3f}")

# --- Aggregate to Economical ---
econ_weights = (w_factors[0]*w_capex +
                w_factors[1]*w_opex +
                w_factors[2]*w_lcoe)

ranking = pd.DataFrame({
    "Alternative": alts,
    "CAPEX_w": w_capex,
    "OPEX_w": w_opex,
    "LCOE_w": w_lcoe,
    "Economical": econ_weights
})
ranking["Rank"] = ranking["Economical"].rank(ascending=False, method="dense").astype(int)
ranking = ranking.sort_values("Economical", ascending=False).reset_index(drop=True)

print("\n=== Final Ranking (Economical criterion) ===")
display(ranking)


# In[ ]:




