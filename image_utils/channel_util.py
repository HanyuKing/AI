
def cmyk_to_rgb(c, m, y, k):
    # c,m,y,k: 0~1
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)
    return int(r), int(g), int(b)

def rgb_to_cmyk(r, g, b):
    # RGB must be 0~255
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 1  # pure black

    # 1. Normalize to 0-1
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0

    # 2. Preliminary CMY'
    c_ = 1 - r_
    m_ = 1 - g_
    y_ = 1 - b_

    # 3. Black key
    k = min(c_, m_, y_)

    # 4. Final CMY
    c = (c_ - k) / (1 - k)
    m = (m_ - k) / (1 - k)
    y = (y_ - k) / (1 - k)

    return c, m, y, k
