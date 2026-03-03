with open("app_map.py", "r") as f:
    lines = f.readlines()

# Find bounds
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "# SECTION 2: PLOTLY ICON TIMELINE" in line:
        start_idx = i - 1  # include the line before it which is ====
    if "# Right: AI Assistant with BERT" in line:
        end_idx = i - 1  # include the line before it which is ----

if start_idx != -1 and end_idx != -1:
    forecast_block = lines[start_idx:end_idx]
    
    # Unindent by 4 spaces
    unindented_block = []
    for line in forecast_block:
        if line.startswith("    "):
            unindented_block.append(line[4:])
        else:
            unindented_block.append(line)
            
    # Wrap in global if
    wrapper_start = [
        "# =============================\n",
        "# GLOBAL 12-HOUR FORECAST\n",
        "# =============================\n",
        "if st.session_state.get('selected_lat') or st.session_state.get('map_city'):\n",
        "    if 'weather_data' in locals() and 'error' not in weather_data:\n"
    ]
    
    # We need to indent the unindented block by 4 spaces again since it's going inside `if` block
    # Actually wait. The block from `lines[start_idx:end_idx]` was indented by 8 spaces (inside `with col1: \n  if "error" not in ...`)
    # Wait, in the original code, `col1` has `if "error" in weather_data: else: cur = parse_current(...)`
    # The whole block is inside the `else:` which is at 4 spaces offset from `with col1:` (which is 0 spaces? Let's check.)
    
    # Let's cleanly handle the indentation. We will make sure the block itself starts at 8 spaces in the new wrapper.
    final_block = []
    final_block.extend(wrapper_start)
    for line in forecast_block:
        final_block.append(line) # keep original 8 spaces indentation (it works perfectly inside two nested if statements!)

    # Remove the old block
    new_lines = lines[:start_idx] + lines[end_idx:]
    
    # Insert before Footer
    footer_idx = -1
    for i, line in enumerate(new_lines):
        if "# Footer" in line:
            footer_idx = i
            break
            
    if footer_idx != -1:
        new_lines = new_lines[:footer_idx] + final_block + ['\n'] + new_lines[footer_idx:]
        
    with open("app_map.py", "w") as f:
        f.writelines(new_lines)
    print("Successfully moved forecast block.")
else:
    print(f"Failed to find indices: start={start_idx}, end={end_idx}")
