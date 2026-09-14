# Rendering and visual inspection

Rendering is on demand and does not edit, step, score, or draw random values.
The existing output is an RGB NumPy array or a PNG file. MCP image access is
[proposed separately](MCP_RENDER_CANVAS.md), not implemented.

## Inspect the example trajectories

From the checkout:

```sh
uv run --locked python scripts/run_examples.py --output-dir artifacts/reward
uv run --locked python scripts/run_examples.py --task two_column --output-dir artifacts/preview
```

The first command replays all fifteen [public trajectories](DATA.md), prints a reward
table, and writes PNG/JSON pairs under `artifacts/reward/<task_name>/`. Each JSON includes
the action trace, final state, and reward breakdown. The command creates output
directories and overwrites matching files. Inspect successful designs alongside
the representative attacks; some visibly bad cases still earn full reward.

## Python APIs

```python
from marketcanvas_env.env import MarketCanvasEnv

env = MarketCanvasEnv(render_mode="rgb_array")
env.reset()
env.execute_action(
    {
        "op": "add_element",
        "element": {
            "type": "text",
            "role": "headline",
            "content": "Summer Sale",
            "x": 100,
            "y": 100,
            "width": 600,
            "height": 80,
        },
    }
)
rgb = env.render()  # Independent uint8 array, shape (600, 800, 3).
env.save_png("/tmp/marketcanvas-preview.png")
```

`render_mode=None` makes `env.render()` return None; `save_png(path)` works in
either mode. Reset is required before image output. Rendering remains available
after termination. The API requires an existing parent directory; the replay
CLI creates it.

For a canonical snapshot, use `render_rgb(state)` or `save_png(state, path)`
from [rendering.py](../src/marketcanvas_env/rendering.py). It accepts the fixed
800×600 canvas and valid uniquely identified elements. Malformed inputs and
unsupported dimensions raise ValueError. Reward evaluation separately supports
defensive scoring of malformed snapshots.

## Drawing rules and limits

- Draw back to front by z-index and stable ID, with half-open rectangle bounds.
  Off-canvas geometry is clipped before allocation. External fractional boxes
  round outward to touched pixels.
- Shapes and image placeholders have solid fills. Image content is metadata;
  there is no photo loading, image generation, or automatic label on placeholders.
- Text elements have transparent backgrounds. Button labels draw on their own fill.
  Other nonempty shape labels follow the shared text-bearing semantics.
- Text uses Pillow's embedded Aileron font, single-line layout, configured inset
  (default four pixels), and left/center/right alignment. Newline text is rejected.
  Font metrics and inset match reward visibility helpers.
- Draw actual antialiased glyphs and clip them to the inset/canvas. Reward uses
  rectangular ink approximations, so pixel visibility and scoring are not identical.
  Contrast lookup still uses whole-box background containment; rendering does not
  repair this [known exploit](REWARD_HACKING.md).
- Reproducibility is scoped to the locked Pillow/font stack. Recheck after dependency
  changes; font licensing/attribution remains with the Pillow distribution.
- No GUI, event loop, external assets, or full-canvas rasterization is needed during
  transitions or reward evaluation.

Tests cover pixel geometry, clipping, stacking, text layout, PNG repeatability,
mutable-array isolation, and state/reward/RNG purity. Gymnasium also checks RGB
output; its absent-FPS/registry warnings are expected for this static environment.
See [development checks](DEVELOPMENT.md).
