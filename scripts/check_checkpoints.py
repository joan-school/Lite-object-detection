import os
import re

EXPERTS = ["kitchen", "display", "climate"]
BASE_PATH = "models/experts"

def find_latest_ckpt(files):
    pattern = re.compile(r"ckpt-(\d+)\.index")
    steps = []

    for f in files:
        match = pattern.match(f)
        if match:
            steps.append(int(match.group(1)))

    if not steps:
        return None

    return max(steps)

def check_expert(expert):
    path = os.path.join(BASE_PATH, expert)

    print(f"\n🔍 Checking: {expert}")

    if not os.path.exists(path):
        print("❌ Folder missing")
        return

    files = os.listdir(path)

    # Check checkpoint file
    if "checkpoint" not in files:
        print("❌ 'checkpoint' file missing")
        return
    else:
        print("✔ checkpoint file exists")

    # Find latest ckpt
    latest = find_latest_ckpt(files)

    if latest is None:
        print("❌ No ckpt-* files found")
        return

    index_file = f"ckpt-{latest}.index"
    data_file = f"ckpt-{latest}.data-00000-of-00001"

    if index_file in files and data_file in files:
        print(f"✔ Latest checkpoint: ckpt-{latest}")
    else:
        print(f"⚠️ Incomplete checkpoint for ckpt-{latest}")

def main():
    print("=== CHECKING ALL EXPERT CHECKPOINTS ===")

    for expert in EXPERTS:
        check_expert(expert)

if __name__ == "__main__":
    main()