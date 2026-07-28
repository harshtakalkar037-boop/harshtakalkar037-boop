"""
Assembles dark.svg / light.svg from the .npy data produced by generate_data.py.
"""
import numpy as np
import random

random.seed(11)
np.random.seed(11)

W, H = 1180, 610
TITLEBAR_H = 32
FRAME_X, FRAME_Y, FRAME_W, FRAME_H = 56, 96, 392, 446
DOT_S = 1.7

PALETTE = {
    "dark": dict(bg="#0A101F", panel="#0E1526", chrome="#22D3EE", chrome_dim="#0891B2",
                 accent="#10B981", portrait="#A78BFA", text="#94A3B8", value="#F8FAFC",
                 border="#1E2A44", red="#F87171"),
    "light": dict(bg="#F8FAFC", panel="#EEF2F7", chrome="#0891B2", chrome_dim="#0E7490",
                  accent="#059669", portrait="#7C3AED", text="#64748B", value="#0F172A",
                  border="#CBD5E1", red="#DC2626"),
}

FIELDS = [
    ("Subject", "Harsh Takalkar"),
    ("Role", "Machine Learning / Embedded Systems"),
    ("Origin", "Pune, India"),
    ("Education", "B.Tech ECE, PICT"),
    ("Status", "Training Models + Building Hardware"),
    ("ToolChain", "VS Code, Git, Arduino IDE, Jupyter"),
    ("Core.Lang", "Python, C++, Java, SQL"),
    ("Core.Frontend", "HTML, CSS, JavaScript"),
    ("Core.Backend", "SQL Systems"),
    ("Core.Database", "MySQL"),
    ("Core.Infra", "Git, GitHub, Vercel"),
    ("Grid.Mail", "harshtakalkar037@gmail.com"),
    ("Grid.Portfolio", "harsh58.onrender.com"),
    ("Grid.LinkedIn", "linkedin.com/in/harsh-takalkar"),
    ("Grid.GitHub", "harshtakalkar037-boop"),
]
HANDLE = "@harshtakalkar037-boop"

INTRO_DUR = 3.2
PORTRAIT_DUR = 3.0
TRANS_DUR = 1.3
LOGO_DUR = 2.0
LOOP_DUR = PORTRAIT_DUR + TRANS_DUR + LOGO_DUR + TRANS_DUR + LOGO_DUR + TRANS_DUR + LOGO_DUR + TRANS_DUR


def r1(v):
    return f"{v:.1f}".rstrip("0").rstrip(".") if "." in f"{v:.1f}" else f"{v:.1f}"


def dots_path(xs, ys, s=DOT_S):
    """One compound <path> for many square dots -- keeps element count sane."""
    parts = []
    for x, y in zip(xs, ys):
        parts.append(f"M{r1(x)} {r1(y)}h{r1(s)}v{r1(s)}h-{r1(s)}z")
    return "".join(parts)


def build(theme):
    pal = PALETTE[theme]
    is_dark = theme == "dark"

    px = np.load("portrait_px.npy")
    py = np.load("portrait_py.npy")
    is_fg = np.load("portrait_is_fg.npy")

    if is_dark:
        keep = is_fg
    else:
        keep = np.ones_like(is_fg, dtype=bool)

    idx = np.nonzero(keep)[0]
    n = len(idx)

    # ---- intro groups (scattered, not spatial) ----
    intro_order = idx.copy()
    np.random.shuffle(intro_order)
    N_INTRO = 60
    intro_groups = np.array_split(intro_order, N_INTRO)

    # ---- drift bands (scattered assignment + centroid drift toward glyph 1) ----
    N_BANDS = 24
    band_order = idx.copy()
    np.random.shuffle(band_order)
    bands = np.array_split(band_order, N_BANDS)

    g1 = np.load("trav_g1.npy")
    logo1_centroid = (g1[0].mean(), g1[1].mean())

    svg_parts = []

    # intro + drift dot groups
    for bi, band in enumerate(bands):
        bx, by = px[band], py[band]
        path_d = dots_path(bx, by)
        # stagger intro start across bands, spread over ~0..1.9s so all
        # settle well before the 3.2s intro ends
        intro_begin = (bi / N_BANDS) * 1.9 + np.random.uniform(0, 0.15)
        dx = (logo1_centroid[0] - bx.mean()) * 0.10
        dy = (logo1_centroid[1] - by.mean()) * 0.10
        drift_begin = INTRO_DUR + np.random.uniform(0, 0.4)
        svg_parts.append(
            f'<g fill="{pal["portrait"]}" opacity="0">'
            f'<animate attributeName="opacity" to="1" begin="{r1(intro_begin)}s" dur="0.9s" fill="freeze"/>'
            f'<g>'
            f'<animateTransform attributeName="transform" type="translate" '
            f'values="0,0;{r1(dx)},{r1(dy)};0,0" keyTimes="0;0.45;1" '
            f'dur="{r1(LOOP_DUR)}s" begin="{r1(drift_begin)}s" repeatCount="indefinite"/>'
            f'<path d="{path_d}"/>'
            f"</g></g>"
        )

    # traveler swarm
    start = np.load("trav_start.npy")
    g1 = np.load("trav_g1.npy")
    g2 = np.load("trav_g2.npy")
    g3 = np.load("trav_g3.npy")

    t = [0, PORTRAIT_DUR, PORTRAIT_DUR + TRANS_DUR,
         PORTRAIT_DUR + TRANS_DUR + LOGO_DUR,
         PORTRAIT_DUR + 2 * TRANS_DUR + LOGO_DUR,
         PORTRAIT_DUR + 2 * TRANS_DUR + 2 * LOGO_DUR,
         PORTRAIT_DUR + 3 * TRANS_DUR + 2 * LOGO_DUR,
         PORTRAIT_DUR + 3 * TRANS_DUR + 3 * LOGO_DUR,
         LOOP_DUR]
    kt = ";".join(r1(v / LOOP_DUR) for v in t)

    trav_group = [f'<g fill="{pal["chrome"]}">']
    n_trav = start.shape[1]
    for i in range(n_trav):
        sx, sy = start[0, i], start[1, i]
        x1, y1 = g1[0, i], g1[1, i]
        x2, y2 = g2[0, i], g2[1, i]
        x3, y3 = g3[0, i], g3[1, i]
        xs_vals = [sx, sx, x1, x1, x2, x2, x3, x3, sx]
        ys_vals = [sy, sy, y1, y1, y2, y2, y3, y3, sy]
        xv = ";".join(r1(v) for v in xs_vals)
        yv = ";".join(r1(v) for v in ys_vals)
        ov = "0;0;1;1;1;1;1;1;0"
        trav_group.append(
            f'<rect width="1.9" height="1.9" x="{r1(sx)}" y="{r1(sy)}" opacity="0">'
            f'<animate attributeName="x" values="{xv}" keyTimes="{kt}" dur="{r1(LOOP_DUR)}s" begin="{r1(INTRO_DUR)}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y" values="{yv}" keyTimes="{kt}" dur="{r1(LOOP_DUR)}s" begin="{r1(INTRO_DUR)}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{ov}" keyTimes="{kt}" dur="{r1(LOOP_DUR)}s" begin="{r1(INTRO_DUR)}s" repeatCount="indefinite"/>'
            f"</rect>"
        )
    trav_group.append("</g>")
    svg_parts.append("".join(trav_group))

    portrait_layer = "".join(svg_parts)

    # ---- info panel ----
    INFO_X = FRAME_X + FRAME_W + 48
    INFO_TOP = 78
    ROW_H = 23
    LABEL_FS = 14
    HEADER_FS = 13
    rows_svg = []
    for i, (label, value) in enumerate(FIELDS):
        y = INFO_TOP + i * ROW_H
        label_w = len(label) * 7.1
        leader_x0 = INFO_X + label_w + 4
        value_w = len(value) * 6.6
        leader_x1 = W - 48 - value_w - 4
        leader_x1 = max(leader_x1, leader_x0 + 6)
        rows_svg.append(
            f'<text x="{r1(INFO_X)}" y="{r1(y)}" font-size="{LABEL_FS}" fill="{pal["text"]}" font-family="JetBrains Mono, monospace">{label}</text>'
            f'<line x1="{r1(leader_x0)}" y1="{r1(y - 4)}" x2="{r1(leader_x1)}" y2="{r1(y - 4)}" '
            f'stroke="{pal["border"]}" stroke-width="1" stroke-dasharray="1.5,3"/>'
            f'<text x="{r1(W - 48)}" y="{r1(y)}" text-anchor="end" font-size="{LABEL_FS}" '
            f'fill="{pal["value"]}" font-family="JetBrains Mono, monospace" '
            f'textLength="{r1(value_w)}" lengthAdjust="spacingAndGlyphs">{value}</text>'
        )
    info_svg = "".join(rows_svg)

    # ---- header: LIVE badge + handle pill ----
    header_svg = f"""
    <text x="{r1(INFO_X)}" y="32" font-size="{HEADER_FS}" letter-spacing="2"
          fill="{pal['chrome']}" font-family="JetBrains Mono, monospace">SYSTEM.INFO</text>
    <g transform="translate({W-48-118},14)">
      <circle cx="7" cy="7" r="4" fill="{pal['red']}">
        <animate attributeName="opacity" values="1;0.25;1" dur="1.6s" repeatCount="indefinite"/>
      </circle>
      <text x="18" y="11" font-size="11" letter-spacing="1.5" fill="{pal['red']}"
            font-family="JetBrains Mono, monospace">LIVE</text>
    </g>
    <g transform="translate({INFO_X},44)">
      <rect x="0" y="-14" width="{9*len(HANDLE)+16}" height="20" rx="10" fill="{pal['panel']}" stroke="{pal['border']}"/>
      <text x="10" y="0" font-size="14" fill="{pal['accent']}" font-family="JetBrains Mono, monospace">{HANDLE}</text>
    </g>
    """

    # ---- portrait frame chrome ----
    frame_svg = f"""
    <text x="{FRAME_X}" y="{FRAME_Y-14}" font-size="13" letter-spacing="2"
          fill="{pal['chrome']}" font-family="JetBrains Mono, monospace">VISUAL.MAP</text>
    <rect x="{FRAME_X-1}" y="{FRAME_Y-1}" width="{FRAME_W+2}" height="{FRAME_H+2}"
          fill="{pal['panel']}" stroke="{pal['border']}" stroke-width="1"/>
    <clipPath id="frameClip-{theme}">
      <rect x="{FRAME_X}" y="{FRAME_Y}" width="{FRAME_W}" height="{FRAME_H}"/>
    </clipPath>
    <g clip-path="url(#frameClip-{theme})" shape-rendering="crispEdges">
      {portrait_layer}
    </g>
    """

    # ---- titlebar ----
    titlebar_svg = f"""
    <rect x="0" y="0" width="{W}" height="{TITLEBAR_H}" fill="{pal['panel']}"/>
    <circle cx="18" cy="16" r="5.5" fill="#F87171"/>
    <circle cx="38" cy="16" r="5.5" fill="#FBBF24"/>
    <circle cx="58" cy="16" r="5.5" fill="#34D399"/>
    <text x="{W/2}" y="20" text-anchor="middle" font-size="12.5" fill="{pal['text']}"
          font-family="JetBrains Mono, monospace">profile.sh --live</text>
    <line x1="0" y1="{TITLEBAR_H}" x2="{W}" y2="{TITLEBAR_H}" stroke="{pal['border']}" stroke-width="1"/>
    """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect x="0" y="0" width="{W}" height="{H}" rx="10" fill="{pal['bg']}"/>
  {titlebar_svg}
  {frame_svg}
  {header_svg}
  {info_svg}
  <rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="10" fill="none" stroke="{pal['border']}" stroke-width="1"/>
</svg>"""
    return svg


for theme in ("dark", "light"):
    svg = build(theme)
    fname = f"{theme}.svg"
    with open(fname, "w") as f:
        f.write(svg)
    import os
    print(theme, f"{os.path.getsize(fname)/1024:.1f} KB")
