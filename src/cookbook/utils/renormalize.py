#!/usr/bin/env python3
import argparse
from decimal import Decimal, getcontext, InvalidOperation
from pathlib import Path
import yaml

# Use high precision for safe normalization math
getcontext().prec = 50

def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data, None

def save_yaml(path: Path, data, yaml_obj=None):
    if yaml_obj is not None:
        yaml_obj.indent(mapping=2, sequence=4, offset=2)
        with path.open("w", encoding="utf-8") as f:
            yaml_obj.dump(data, f)
    else:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

def normalize_ratios(sources):
    ratios = []
    idxs = []
    for i, src in enumerate(sources):
        if "target_ratio" in src:
            try:
                ratios.append(Decimal(str(src["target_ratio"])))
            except (InvalidOperation, TypeError, ValueError):
                raise ValueError(f"Non-numeric target_ratio at sources[{i}]")
            idxs.append(i)

    if not ratios:
        raise ValueError("No target_ratio fields found under dataset.sources.")

    total = sum(ratios)
    if total == 0:
        raise ValueError("Sum of target_ratio is 0; cannot normalize.")

    new_ratios = [r / total for r in ratios]
    return idxs, ratios, new_ratios

def verify(r_old, r_new, tol_sum=Decimal("1e-12"), tol_ratio=Decimal("1e-10")):
    # Sum check
    sum_new = sum(r_new)
    if abs(sum_new - Decimal(1)) > tol_sum:
        return False, f"Sum check failed: sum(new)={sum_new}"

    # Proportionality check: r_new[i]/r_new[j] ≈ r_old[i]/r_old[j]
    # Skip pairs where old ratio is 0.
    for i in range(len(r_old)):
        if r_old[i] == 0:
            continue
        for j in range(i + 1, len(r_old)):
            if r_old[j] == 0:
                continue
            lhs = r_new[i] / r_new[j]
            rhs = r_old[i] / r_old[j]
            if abs(lhs - rhs) > tol_ratio * max(Decimal(1), abs(rhs)):
                return False, (f"Proportionality check failed for pair ({i},{j}): "
                               f"new_ratio={lhs} vs old_ratio={rhs}")
    return True, "Verification passed."

def main():
    ap = argparse.ArgumentParser(description="Renormalize dataset.sources[*].target_ratio to sum to 1.")
    ap.add_argument("yaml_path", type=Path, help="Path to the YAML file to modify in place.")
    ap.add_argument("--no-backup", action="store_true", help="Do not write a .bak backup file.")
    args = ap.parse_args()

    data, yaml_obj = load_yaml(args.yaml_path)

    # Navigate to dataset.sources
    try:
        sources = data["dataset"]["sources"]
        if not isinstance(sources, list):
            raise TypeError
    except Exception:
        raise KeyError("Expected path dataset.sources (a list) in the YAML.")

    idxs, old_ratios, new_ratios = normalize_ratios(sources)

    # Write back the normalized ratios (use float for YAML emission; precise math kept in Decimal)
    for idx, new_r in zip(idxs, new_ratios):
        # Emit with up to 18 significant digits for readability
        sources[idx]["target_ratio"] = float(f"{new_r:.18g}")

    # Backup then save
    if not args.no_backup:
        backup = args.yaml_path.with_suffix(args.yaml_path.suffix + ".bak")
        backup.write_text(args.yaml_path.read_text(encoding="utf-8"), encoding="utf-8")

    save_yaml(args.yaml_path, data, yaml_obj)

    # Reload to verify from disk (ensures what got written still checks out)
    data2, _ = load_yaml(args.yaml_path)
    sources2 = data2["dataset"]["sources"]

    reloaded = []
    for i in idxs:
        try:
            reloaded.append(Decimal(str(sources2[i]["target_ratio"])))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError(f"After save, target_ratio at sources[{i}] is not numeric.")

    ok, msg = verify(old_ratios, reloaded)
    print(msg)

    if ok:
        print("Details:")
        print(f"  Count normalized: {len(new_ratios)}")
        print(f"  Sum(new ratios):  {sum(reloaded):.30f}")
        # Show a few samples
        for i in range(min(5, len(new_ratios))):
            print(f"  sources[{idxs[i]}]: old={old_ratios[i]:.6E} -> new={reloaded[i]:.6E}")

if __name__ == "__main__":
    """
    python src/cookbook/utils/renormalize.py
    """
    main()