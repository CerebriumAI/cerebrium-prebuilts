import os
import subprocess

# get list of directories
dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
dirs = [d for d in dirs if not d.startswith(".")]

blacklist = ["bakllava", "falcon-40b-instruct-quantized", "-gpt-neox-20b"]
dirs = [d for d in dirs if d not in blacklist]
dirs.sort()

new_dirs = []
for d in dirs:
    tmp = d.split("-")[0]
    add = True
    for i in new_dirs:
        if i.startswith(tmp):
            add = False
            break

    if add:
        new_dirs.append(d)

limit = 5
new_dirs = new_dirs[:limit]
info = "\n\t-".join(new_dirs)
print(f"Deploying: \n\t-{info}")

for d in new_dirs:
    os.chdir(d)
    # cerebrium deploy --name prebuilt-{d}
    name = "prebuilt-" + d.replace("_", "-")
    print(f"Deploying {name}")
    subprocess.call(
        [
            "cerebrium",
            "deploy",
            "--name",
            name,
            "--disable-predict"
        ]
    )
    os.chdir("..")
