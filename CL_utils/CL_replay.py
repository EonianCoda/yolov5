from pathlib import Path

FILE = Path("exemplar.txt")
def read_exemplar():
    with open(FILE, "r") as f:
        img_ids = f.readlines()
    return img_ids