import importlib.metadata as md
import sys

print("python", sys.version.split()[0])
for ten in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch", "Pillow"]:
    try:
        print("%-14s %s " % (ten, md.version(ten)))
    except Exception:
        print("%-14s THIEU" %ten)