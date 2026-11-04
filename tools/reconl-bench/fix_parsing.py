import re

path = "tools/reconl-bench/run_resolution_sweep.sh"
with open(path, "r") as f:
    lines = f.readlines()

# Replace lines 151-155 (0-indexed: 150-154) with new parsing code
new_parsing = [
    '    fps_str=$(echo "$output" | sed -n \'s/.*(\\([0-9.]*\\) fps at the mean).*/\\1/p\')\n',
    '    fps_str=${fps_str:-NA}\n',
    '    total_str=$(echo "$output" | sed -n \'s/.*avg \\([0-9.]*\\).*/\\1/p\')\n',
    '    total_str=${total_str:-NA}\n',
    '    shadow_str=$(echo "$output" | sed -n \'s/.*pass \\(—\\|[^[:space:]]*\\).*/\\1/p\' | head -1)\n',
    '    shadow_str=${shadow_str:-NA}\n',
    '    alloc_str=$(echo "$output" | sed -n \'s/.*(\\([0-9.]*\\) per frame).*/\\1/p\')\n',
    '    alloc_str=${alloc_str:-NA}\n',
    '    final_tier=$(echo "$output" | sed -n \'s/.*tier \\([^[:space:]]*\\).*/\\1/p\')\n',
    '    final_tier=${final_tier:-NA}\n',
]

# Lines 151-155 are indices 150-154 (the 5 sed lines)
# Keep line 150 (local declaration) and replace 151-155
result = lines[:150] + new_parsing + lines[155:]

with open(path, "w") as f:
    f.writelines(result)

print("Parsing section replaced successfully")
print(f"Old lines 150-155 replaced with {len(new_parsing)} new lines")
