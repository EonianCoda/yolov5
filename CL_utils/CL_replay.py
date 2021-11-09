from pathlib import Path

FILE = Path("exemplar.txt")
def read_exemplar():
    with open(FILE, "r") as f:
        img_ids = [int(img_id) for img_id in f.readlines()]
    return img_ids