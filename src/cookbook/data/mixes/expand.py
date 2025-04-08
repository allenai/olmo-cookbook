import subprocess
from pathlib import Path

input_file = "datadelve_dclm_sample.txt"      # Input file with s3 paths (with wildcards)
output_file = "datadelve_dclm_sample_expanded.txt"

def list_s3_files(s3_prefix):
    result = subprocess.run(
        ["aws", "s3", "ls", s3_prefix, "--recursive"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"❌ Error with path: {s3_prefix}")
        return []

    files = []
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 4:
            file_key = " ".join(parts[3:])
            if file_key.endswith(".npy"):  # Only include .npy files
                full_path = f"s3://ai2-llm/{file_key}"
                files.append(full_path)
    return files

def extract_prefix(path):
    prefix = path.split("**")[0]
    return prefix if prefix.endswith("/") else prefix + "/"

all_entries = []

with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for line in lines:
    print(line)
    prefix = extract_prefix(line)
    files = list_s3_files(prefix)
    if files:
        header = f"#SOURCE: {prefix} ({len(files)} files)"
        all_entries.append(header)
        all_entries.extend(sorted(files))  # Optional: sort if you want consistency

with open(output_file, "w") as out:
    out.write("\n".join(all_entries) + "\n")

print(f"✅ Done! Expanded paths written to: {output_file}")
