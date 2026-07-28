# Banner source

Regenerate `dark.svg` / `light.svg` from here — don't hand-edit the SVGs.

```
pip install pillow numpy scipy opencv-python-headless
python3 generate_data.py   # rebuilds the .npy dot/traveler data from crop.png + fg_mask.npy
python3 build_svg.py       # assembles dark.svg and light.svg from that data
```

- `crop.png` — the head-and-shoulders crop used for the portrait
- `fg_mask.npy` — GrabCut foreground mask (dark-mode subject silhouette)
- `glyph_neural.png` / `glyph_chip.png` / `glyph_code.png` — the three shapes the traveler dots morph between
- To change info-panel text, palette, or timing, edit the config block at the top of each script
