from rembg import new_session
try:
    # Try to load birefnet-portrait which is SOTA for portraits
    session = new_session("birefnet-portrait")
    print("SUCCESS: birefnet-portrait is supported")
except Exception as e:
    print(f"FAILURE: {e}")

try:
    # Fallback to u2net_human_seg just to be sure
    session = new_session("u2net_human_seg")
    print("SUCCESS: u2net_human_seg is supported")
except Exception as e:
    print(f"FAILURE: {e}")
