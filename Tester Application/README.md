# ReconL Tester

An interactive test scene for the ReconL renderer. A walled courtyard with walls,
pillars and a ring of squares, circles and triangles, lit by a low sun so every
object throws a shadow. Reflections are real geometry, drawn mirrored under a
translucent floor.

```
cargo run --release -p reconl-tester
cargo run --release -p reconl-tester -- --backend=vulkan
cargo run --release -p reconl-tester -- --backend=vulkan --res=2K
```

## Controls

| Key | Does |
|---|---|
| `W` `A` `S` `D` | Walk |
| `Space` / `Left Shift` | Rise / descend |
| Arrow keys | Look |
| Mouse drag | Look |
| `Esc` | Open or close the settings menu |
| `Tab` | Show or hide the statistics |
| `1` `2` `3` `4` | Shadow level shortcut |
| `Q` / `E` | Reflection level shortcut |
| `R` | Resolution shortcut |

In the menu, `Up`/`Down` selects an item, `Left`/`Right` changes it, and `Esc`
closes it. `Quit` is the last item.

## Lighting

Two lights, because one cannot make a shadow that looks like shade:

* **The sun**, 35° above the horizon, casting shadows. A 5.2-unit pillar throws a
  7.4-unit shadow.
* **A sky fill**, shadowless, from the opposite side, at 42% intensity and a
  slight blue.

The engine's ambient term is a fixed 6% and no ABI field sets it, so a scene lit
by one sun has shadows at 6% of its albedo — black. The fill light is how a scene
without an ambient control gets light into its shadows, and it is the difference
between a shadow and a hole. If you compare this against the golden images, that
is the first thing that differs: they are lit by one light on purpose.

## What the settings do

**Shadows** move one ladder: cascades, texel budget and filter together.

| Level | Cascades | Requested | Granted on this machine |
|---|---|---|---|
| Light | 1 | 2 MiB | 512×512 ×1 HARD |
| Medium | 2 | 8 MiB | 512×512 ×2 PCF3 |
| High | 3 | 24 MiB | 1024×1024 ×3 PCF5 |
| Max | 3 | 96 MiB | 2048×2048 ×3 PCF5 |

The budget is what is *requested*; the tier scales it (0.5 on `GpuShared`, which
is what an integrated GPU is read as), caps the cascades (3) and clamps the
filter (PCF5, so PCSS-lite is downgraded). Max used to request 40 MiB, which
rounded down to the same 1024 map as High — two labels, one map. 12 MiB effective
buys exactly 1024 per cascade, and the next power of two needs four times that.

The HUD prints what the device *granted* next to the level you picked, because
the clamp is silent by design and "Max" is a request.

**Reflections** select how much of the world is mirrored: off, the shapes only,
then pillars, then walls too. With reflections on the floor is drawn translucent
(x3 translucent, alpha 0.70) over the mirrored copies.

Each reflected draw uses `REFLECT_BASE + (chunk - BOX)`: a copy of the mesh with
its normals' Y negated. The mirror matrix has a negative determinant and flips a
normal's Y along with a position's, which lights a reflection from below and
renders it black. Negating Y in the mesh cancels that at draw time, so a
reflection is shaded exactly like the object it reflects and the floor is the
only thing dimming it.

**Resolution** re-renders the scene at the chosen size and scales it into a fixed
1280×720 window, so the window does not move while you are comparing.

## Statistics

| Line | Means |
|---|---|
| `FPS AVG` | Mean frame rate over the last 240 frames |
| `1% LOW` | Frame rate during the slowest 1% of those frames |
| `10% LOW` | Frame rate during the slowest 10% |
| `PEAK` | Best single frame |

The lows are the frames that decide whether a scene feels smooth; an average
hides them. Changing resolution or shadow level clears the history, because a
statistic that mixes two configurations describes neither. Frames rendered while
the menu is open are not recorded.

## Headless checks

```
reconl-tester --shot=frame.png --backend=vulkan --res=720P --reflections=3
reconl-tester --check-camera
```

`--shot` renders one frame, writes it as a PNG and prints the shadow map the
device granted, which is how the image is checked without a window.
`--check-camera` prints the tester's view matrix beside the host's reference
pose, so the free camera can be shown to agree with the matrix the golden images
were rendered through (they differ by 2×10⁻⁶).

## Notes

The frame submission lives in `src/gfx.rs` rather than using
`reconl_host::frame::Renderer`. That renderer builds one pipeline, which is right
for a benchmark that times a single configuration; reflections need a second,
alpha pipeline for the mirrored copies and the floor drawn over them, and a
pipeline cannot be switched without owning the command list.

The camera is owned here for the same reason: `CameraSpec` describes one fixed
pose, and a scene you can walk around needs an arbitrary eye.
