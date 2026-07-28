"""
GitHub profile banner generator.
Two-layer architecture: dense static/drifting portrait (dithered dots) +
sparse traveler swarm that morphs between three glyphs.
Source of truth: this script + the .npy data it reads. Regenerate the SVGs
from here, don't hand-edit them.
"""
import json
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.optimize import linear_sum_assignment
from scipy import ndimage

random.seed(7)
np.random.seed(7)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
W, H = 1180, 610
TITLEBAR_H = 32

PALETTE = {
    "portrait_dark": "#A78BFA",
    "portrait_light": "#7C3AED",
    "chrome": "#22D3EE",
    "chrome_dim": "#0891B2",
    "accent": "#10B981",
    "bg_dark": "#0A101F",
    "bg_light": "#F8FAFC",
    "panel_dark": "#0E1526",
    "panel_light": "#EEF2F7",
    "text_dark": "#94A3B8",
    "text_light": "#475569",
    "value_dark": "#F8FAFC",
    "value_light": "#0F172A",
}

FIELDS = {
    "Subject": "Harsh Takalkar",
    "Role": "Machine Learning / Embedded Systems",
    "Origin": "Pune, India",
    "Education": "B.Tech ECE, PICT",
    "Status": "Training Models + Building Hardware",
    "ToolChain": "VS Code, Git, Arduino IDE, Jupyter",
    "Core.Lang": "Python, C++, Java, SQL",
    "Core.Frontend": "HTML, CSS, JavaScript",
    "Core.Backend": "SQL Systems",
    "Core.Database": "MySQL",
    "Core.Infra": "Git, GitHub, Vercel",
    "Grid.Mail": "harshtakalkar037@gmail.com",
    "Grid.Portfolio": "harsh58.onrender.com",
    "Grid.LinkedIn": "linkedin.com/in/harsh-takalkar",
    "Grid.GitHub": "harshtakalkar037-boop",
}
HANDLE = "harshtakalkar037-boop"

# loop timeline (seconds)
INTRO_DUR = 3.2
PORTRAIT_DUR = 3.0
TRANS_DUR = 1.3
LOGO_DUR = 2.0
LOOP_DUR = PORTRAIT_DUR + TRANS_DUR + LOGO_DUR + TRANS_DUR + LOGO_DUR + TRANS_DUR + LOGO_DUR + TRANS_DUR
assert abs(LOOP_DUR - 14.2) < 0.01, LOOP_DUR

N_TRAVELERS = 560
N_INTRO_GROUPS = 60
N_DRIFT_BANDS = 24

# ---------------------------------------------------------------------------
# 1. Portrait: tone-map + dither at a resolution tuned for ~17k ink dots
# ---------------------------------------------------------------------------
GRID_W, GRID_H = 168, 191

crop = Image.open("crop.png").convert("L").resize((GRID_W, GRID_H), Image.LANCZOS)
crop = ImageOps.autocontrast(crop, cutoff=1)
crop = crop.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
crop = ImageEnhance.Contrast(crop).enhance(1.3)
dithered = crop.convert("1", dither=Image.FLOYDSTEINBERG)
ink = ~np.array(dithered)  # True = dot present (dark tone)

fg_full = np.load("fg_mask.npy")
fg_img = Image.fromarray((fg_full * 255).astype("uint8")).resize((GRID_W, GRID_H), Image.NEAREST)
fg = np.array(fg_img) > 127

ys, xs = np.nonzero(ink)
n_dots = len(xs)
print(f"portrait ink dots: {n_dots}")

is_fg = fg[ys, xs]  # per-dot: is this dot within the segmented subject?

# ---------------------------------------------------------------------------
# 2. Frame geometry (portrait dot coords -> screen px)
# ---------------------------------------------------------------------------
FRAME_X, FRAME_Y, FRAME_W, FRAME_H = 56, 96, 392, 446
pitch_x = FRAME_W / GRID_W
pitch_y = FRAME_H / GRID_H
dot_s = 1.7

px = FRAME_X + xs * pitch_x
py = FRAME_Y + ys * pitch_y

np.save("portrait_px.npy", px)
np.save("portrait_py.npy", py)
np.save("portrait_is_fg.npy", is_fg)

# ---------------------------------------------------------------------------
# 3. Glyph point clouds (sampled from rasterized glyphs), mapped into the
#    same screen-space frame so travelers can morph in place.
# ---------------------------------------------------------------------------
def glyph_points(path, n):
    im = np.array(Image.open(path))
    ys_g, xs_g = np.nonzero(im > 127)
    idx = np.random.choice(len(xs_g), size=min(n, len(xs_g)), replace=False)
    gx, gy = xs_g[idx], ys_g[idx]
    # normalize to 0..1 within the glyph canvas, then map into the frame box
    gx = gx / im.shape[1]
    gy = gy / im.shape[0]
    sx = FRAME_X + gx * FRAME_W
    sy = FRAME_Y + gy * FRAME_H
    return sx, sy

# ---------------------------------------------------------------------------
# 4. Traveler swarm: sample start positions from the portrait, then compute
#    optimal (Hungarian) assignment into each glyph in turn -> shortest total
#    travel, no crossing artifacts from naive index pairing.
# ---------------------------------------------------------------------------
trav_idx = np.random.choice(n_dots, size=N_TRAVELERS, replace=False)
start_x, start_y = px[trav_idx], py[trav_idx]

g1x, g1y = glyph_points("glyph_neural.png", N_TRAVELERS)
g2x, g2y = glyph_points("glyph_chip.png", N_TRAVELERS)
g3x, g3y = glyph_points("glyph_code.png", N_TRAVELERS)

def match(sx, sy, tx, ty):
    """Hungarian-assign (sx,sy) -> (tx,ty) minimizing total squared distance."""
    n = min(len(sx), len(tx))
    cost = (sx[:n, None] - tx[None, :n]) ** 2 + (sy[:n, None] - ty[None, :n]) ** 2
    row, col = linear_sum_assignment(cost)
    return tx[col], ty[col]

n = min(N_TRAVELERS, len(g1x), len(g2x), len(g3x))
start_x, start_y = start_x[:n], start_y[:n]
g1x, g1y = match(start_x, start_y, g1x, g1y)
g2x, g2y = match(g1x, g1y, g2x, g2y)
g3x, g3y = match(g2x, g2y, g3x, g3y)

# small per-dot jitter on every hop to avoid a perfectly rigid look
def jitter(x, y, sigma=2.2):
    return x + np.random.normal(0, sigma, len(x)), y + np.random.normal(0, sigma, len(y))

g1x, g1y = jitter(g1x, g1y)
g2x, g2y = jitter(g2x, g2y)
g3x, g3y = jitter(g3x, g3y)

np.save("trav_start.npy", np.stack([start_x, start_y]))
np.save("trav_g1.npy", np.stack([g1x, g1y]))
np.save("trav_g2.npy", np.stack([g2x, g2y]))
np.save("trav_g3.npy", np.stack([g3x, g3y]))

print("Phase 1 data ready:", n_dots, "portrait dots,", n, "travelers")
