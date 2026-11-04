//! ReconL Tester — an interactive test scene for the renderer.
//!
//! A walled courtyard with walls, pillars and a ring of squares, circles and
//! triangles, lit by a low sun so everything throws a shadow. Behind the settings
//! are the two things this exists to exercise: the shadow system and reflected
//! geometry.
//!
//! Reflections are built here rather than in the engine. The engine has no
//! reflection pass, and it does not need one to be tested: a reflection is the
//! same geometry drawn with a mirrored model matrix, before a translucent floor
//! that lets it through. That is what a game does when it has no spare render
//! target, and it exercises the same draw path the rest of the scene uses.

mod font;
mod gfx;
mod scene;

use gfx::{Draw, Gfx, Pass, View};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use reconl::abi;
use reconl_host::device::{Config, Device};
use reconl_host::scene::{Scene, ShadowSpec};
use scene::{World, TAG_DISC, TAG_PILLAR, TAG_SHAPE, TAG_WALL};
use std::time::Instant;

/// The window is a fixed size and the scene is scaled into it, so switching
/// resolution changes what the renderer does and not the user's window.
const WINDOW_W: usize = 1280;
const WINDOW_H: usize = 720;
/// The HUD's width.
const PANEL_W: usize = 410;

const RESOLUTIONS: [(&str, u32, u32); 4] = [
    ("2K", 2560, 1440),
    ("1080P", 1920, 1080),
    ("920P", 1632, 920),
    ("720P", 1280, 720),
];

const SHADOW_LEVELS: [&str; 4] = ["LIGHT", "MEDIUM", "HIGH", "MAX"];
const REFLECTION_LEVELS: [&str; 4] = ["OFF", "LOW", "MEDIUM", "MAX"];

fn shadow_spec(level: usize) -> ShadowSpec {
    // Level to (cascades, texel budget, filter).
    //
    // The budget is the *requested* total and the library partitions and scales it
    // per tier, so the map a level actually receives is reported by the device
    // rather than assumed here. The filter ladder is hard, PCF 3x3, PCF 5x5,
    // PCSS-lite.
    // The budgets are sized so the *granted* maps step 512 -> 512 -> 1024 -> 2048
    // on a tier that halves the budget (GpuShared, which is what this machine is
    // read as): 12 MiB effective buys exactly 1024 per cascade, and the next
    // power of two needs 4x that. A 40 MiB request rounded down to 1024 as well,
    // so High and Max used to be the same map with different labels.
    let (cascades, budget, filter) = match level {
        0 => (1u32, 2u64 << 20, 0u32),
        1 => (2, 8 << 20, 1),
        2 => (3, 24 << 20, 2),
        _ => (3, 96 << 20, 3),
    };
    ShadowSpec::custom(cascades, budget, filter)
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// A free camera: a position and a direction, with no fixed look-at target.
///
/// The basis is built the way `reconl_host::scene::CameraSpec` builds its own -
/// `right = normalize(world_up x back)`, `up = back x right`, and the view array
/// stores the three basis vectors as its first three groups of four - so a
/// camera placed where `CameraSpec` sits produces the same matrix, which is what
/// `--check-camera` verifies.
struct Camera {
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len > 1e-6 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

impl Camera {
    /// The pose the reference scene uses, so the tester opens on the same view.
    fn reference() -> Camera {
        let (height, distance) = (12.0f32, 13.0f32);
        let len = (height * height + distance * distance).sqrt();
        Camera {
            pos: [0.0, height, distance],
            yaw: 0.0,
            pitch: (height / len).asin(),
        }
    }

    /// Where the app opens: low enough that the floor is seen at a shallow angle
    /// and the shadows show their length rather than being foreshortened.
    fn opening() -> Camera {
        Camera { pos: [0.0, 9.5, 15.0], yaw: 0.0, pitch: 0.40 }
    }

    /// right, up, back.
    fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let back = [cp * sy, sp, cp * cy];
        let right = normalize(cross([0.0, 1.0, 0.0], back));
        let up = cross(back, right);
        (right, up, back)
    }

    fn view(&self) -> [f32; 16] {
        let (r, u, b) = self.basis();
        let e = self.pos;
        [
            r[0], u[0], b[0], 0.0, //
            r[1], u[1], b[1], 0.0, //
            r[2], u[2], b[2], 0.0, //
            -dot(r, e), -dot(u, e), -dot(b, e), 1.0,
        ]
    }

    fn forward(&self) -> [f32; 3] {
        let (_, _, b) = self.basis();
        [-b[0], -b[1], -b[2]]
    }

    fn right(&self) -> [f32; 3] {
        self.basis().0
    }

    fn move_by(&mut self, direction: [f32; 3], distance: f32) {
        self.pos[0] += direction[0] * distance;
        self.pos[1] += direction[1] * distance;
        self.pos[2] += direction[2] * distance;
        // Keep the eye above the floor and below the sky.
        self.pos[1] = self.pos[1].clamp(0.6, 60.0);
    }

    fn look(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        // Just short of straight up and down, where the basis would collapse.
        self.pitch = (self.pitch + dpitch).clamp(-1.45, 1.45);
    }

    fn view_for(&self, width: u32, height: u32, fov_y_deg: f32, near: f32, far: f32) -> View {
        let aspect = width as f32 / height.max(1) as f32;
        View {
            view: self.view(),
            proj: reconl_host::math::perspective_rh_reversed_z(fov_y_deg, aspect, near, far),
            fov_y_deg,
            near,
            far,
        }
    }
}

// ---------------------------------------------------------------------------
// Draw list
// ---------------------------------------------------------------------------

/// Builds the frame's draw list.
///
/// Order matters and is the whole reflection technique: mirrored copies first,
/// then the translucent floor over them, then everything solid. Drawing the floor
/// first would hide the reflections behind it, and drawing nothing translucent
/// would leave them invisible under an opaque plane.
fn build_draws(world: &World, reflection_level: usize) -> Vec<Draw> {
    let mut draws: Vec<Draw> = Vec::with_capacity(world.draws.len() * 2 + 2);
    if reflection_level == 0 {
        draws.push(scene::floor_opaque());
    } else {
        for draw in &world.draws {
            let reflect = match reflection_level {
                1 => matches!(draw.tag, TAG_DISC | TAG_SHAPE),
                2 => matches!(draw.tag, TAG_DISC | TAG_SHAPE | TAG_PILLAR),
                _ => matches!(draw.tag, TAG_DISC | TAG_SHAPE | TAG_PILLAR | TAG_WALL),
            };
            if !reflect {
                continue;
            }
            draws.push(Draw {
                // The reflected copy of the mesh, not the mesh: see `mirrored`.
                chunk: if (scene::BOX..=scene::SQUARE).contains(&draw.chunk) {
                    scene::REFLECT_BASE + (draw.chunk - scene::BOX)
                } else {
                    draw.chunk
                },
                model: reconl_host::math::mul(&scene::MIRROR_Y, &draw.model),
                pass: Pass::Alpha,
                tag: draw.tag,
            });
        }
        draws.push(scene::floor_glass());
    }
    draws.extend(world.draws.iter().copied());
    draws
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Rolling frame-time statistics.
///
/// "1% low" is the frame rate you get in the worst 1% of frames: the slowest
/// frames averaged and inverted. Quoting it as an fps figure is what makes it
/// comparable with the average, and it is the number that catches a stutter an
/// average hides.
struct Stats {
    times: Vec<f64>,
    capacity: usize,
}

impl Stats {
    fn new(capacity: usize) -> Stats {
        Stats { times: Vec::with_capacity(capacity), capacity }
    }

    fn push(&mut self, ms: f64) {
        if self.times.len() == self.capacity {
            self.times.remove(0);
        }
        self.times.push(ms);
    }

    fn clear(&mut self) {
        self.times.clear();
    }

    fn average(&self) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        1000.0 * self.times.len() as f64 / self.times.iter().sum::<f64>()
    }

    fn peak(&self) -> f64 {
        let fastest = self.times.iter().cloned().fold(f64::INFINITY, f64::min);
        if fastest.is_finite() && fastest > 0.0 {
            1000.0 / fastest
        } else {
            0.0
        }
    }

    fn low(&self, percent: f64) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        let mut sorted = self.times.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        let count = ((sorted.len() as f64) * percent / 100.0).ceil().max(1.0) as usize;
        let count = count.min(sorted.len());
        let mean = sorted[..count].iter().sum::<f64>() / count as f64;
        if mean > 0.0 {
            1000.0 / mean
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

struct State {
    shadow: usize,
    reflection: usize,
    resolution: usize,
    show_stats: bool,
}

/// The in-game settings menu. Nothing is applied to the renderer here: the menu
/// edits `State`, and the frame loop notices what changed, the same path the
/// keyboard shortcuts use.
struct Menu {
    open: bool,
    index: usize,
}

const MENU_ITEMS: [&str; 5] = ["SHADOWS", "REFLECTIONS", "RESOLUTION", "SHOW STATS", "QUIT"];

/// What the frame loop must redo after the settings changed.
#[derive(Default)]
struct Changes {
    shadow: bool,
    reflection: bool,
    resolution: bool,
    quit: bool,
}

impl Menu {
    fn value(&self, state: &State) -> String {
        match self.index {
            0 => SHADOW_LEVELS[state.shadow].to_string(),
            1 => REFLECTION_LEVELS[state.reflection].to_string(),
            2 => RESOLUTIONS[state.resolution].0.to_string(),
            3 => if state.show_stats { "ON" } else { "OFF" }.to_string(),
            _ => String::new(),
        }
    }

    /// Steps the selected item. `delta` is -1 or +1.
    fn adjust(&mut self, state: &mut State, delta: i32) -> Changes {
        let mut changes = Changes::default();
        match self.index {
            0 => {
                let next = (state.shadow as i32 + delta).clamp(0, 3) as usize;
                if next != state.shadow {
                    state.shadow = next;
                    changes.shadow = true;
                }
            }
            1 => {
                let next = (state.reflection as i32 + delta).clamp(0, 3) as usize;
                if next != state.reflection {
                    state.reflection = next;
                    changes.reflection = true;
                }
            }
            2 => {
                let count = RESOLUTIONS.len() as i32;
                let next = ((state.resolution as i32 + delta).rem_euclid(count)) as usize;
                if next != state.resolution {
                    state.resolution = next;
                    changes.resolution = true;
                }
            }
            3 => {
                state.show_stats = !state.show_stats;
            }
            _ => changes.quit = delta > 0,
        }
        changes
    }
}

// ---------------------------------------------------------------------------
// Drawing the overlay
// ---------------------------------------------------------------------------

fn draw_hud(pixels: &mut [u8], width: u32, height: u32, state: &State, stats: &Stats, shadow_map: &str) {
    let scale = 2;
    let line = 7 * scale + 5;
    let pad = 10;
    let rows = if state.show_stats { 9 } else { 6 };
    font::rect(pixels, width, height, 10, 10, PANEL_W as i32, pad * 2 + rows * line, [8, 10, 14, 205]);

    let white = [235, 238, 245, 255];
    let accent = [120, 210, 255, 255];
    let good = [140, 225, 155, 255];

    let mut y = 10 + pad;
    font::text(pixels, width, height, 20, y, scale, "RECONL TESTER", accent);
    y += line;

    let (name, rw, rh) = RESOLUTIONS[state.resolution];
    let lines = [
        format!("SHADOWS  {}  {}", SHADOW_LEVELS[state.shadow], shadow_map),
        format!("REFLECT  {}", REFLECTION_LEVELS[state.reflection]),
        format!("RES      {} {}X{}", name, rw, rh),
    ];
    for text in &lines {
        font::text(pixels, width, height, 20, y, scale, text, white);
        y += line;
    }

    if !state.show_stats {
        return;
    }
    y += 4;
    font::rect(pixels, width, height, 18, y, PANEL_W as i32 - 16, 1, [90, 100, 120, 255]);
    y += 8;

    font::text(pixels, width, height, 20, y, scale, &format!("FPS AVG   {:.1}", stats.average()), white);
    y += line;
    font::text(pixels, width, height, 20, y, scale, &format!("1% LOW    {:.1}", stats.low(1.0)), good);
    y += line;
    font::text(pixels, width, height, 20, y, scale, &format!("10% LOW   {:.1}", stats.low(10.0)), good);
    y += line;
    font::text(pixels, width, height, 20, y, scale, &format!("PEAK      {:.1}", stats.peak()), white);
}

fn draw_menu(pixels: &mut [u8], width: u32, height: u32, state: &State, menu: &Menu) {
    let scale = 3;
    let line = 7 * scale + 9;
    let box_w = 470;
    let box_h = 96 + MENU_ITEMS.len() as i32 * line;
    let x0 = (width as i32 - box_w) / 2;
    let y0 = (height as i32 - box_h) / 2;

    let border = [110, 190, 235, 255];
    let panel = [12, 15, 21, 240];
    font::rect(pixels, width, height, x0, y0, box_w, box_h, border);
    font::rect(pixels, width, height, x0 + 2, y0 + 2, box_w - 4, box_h - 4, panel);

    let white = [235, 238, 245, 255];
    let dim = [130, 138, 152, 255];
    let accent = [120, 210, 255, 255];

    let mut y = y0 + 18;
    font::text(pixels, width, height, x0 + 22, y, scale, "SETTINGS", accent);
    y += line + 8;

    for (index, label) in MENU_ITEMS.iter().enumerate() {
        let selected = index == menu.index;
        if selected {
            font::rect(pixels, width, height, x0 + 12, y - 4, box_w - 24, line - 2, [34, 52, 72, 255]);
        }
        let colour = if selected { white } else { dim };
        font::text(pixels, width, height, x0 + 22, y, scale, label, colour);
        let value = if *label == "QUIT" {
            "ENTER".to_string()
        } else {
            menu.value(state)
        };
        let value_colour = if selected { accent } else { dim };
        let vx = x0 + box_w - 22 - font::measure(&value, scale);
        font::text(pixels, width, height, vx, y, scale, &value, value_colour);
        y += line;
    }

    y += 6;
    font::text(
        pixels,
        width,
        height,
        x0 + 22,
        y,
        2,
        "UP/DOWN SELECT   LEFT/RIGHT CHANGE   ESC CLOSE",
        dim,
    );
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// What the device says the shadow map really is.
///
/// The requested budget is not the granted one: the tier scales it, caps the
/// cascade count and clamps the filter, and every clamp is silent by design. The
/// only honest answer to "what am I looking at" is the device's own report.
fn shadow_info(device: &Device) -> String {
    match device.stats() {
        Ok(stats) => {
            let filter = match stats.shadows.filter_active {
                0 => "HARD",
                1 => "PCF3",
                2 => "PCF5",
                _ => "PCSS",
            };
            format!(
                "{}X{} X{} {}",
                stats.shadows.map_width, stats.shadows.map_height,
                stats.shadows.cascades_active, filter
            )
        }
        Err(e) => format!("unknown ({e})"),
    }
}

fn parse_backend(name: &str) -> u32 {
    match name {
        "vulkan" => abi::backend::VULKAN,
        "d3d11" | "dx11" => abi::backend::D3D11,
        "soft-cpu" | "cpu" => abi::backend::SOFT_CPU,
        _ => abi::backend::NONE,
    }
}

fn run() -> Result<(), String> {
    let mut backend = abi::backend::NONE;
    let mut start_res = 1usize;
    let mut start_shadow = 2usize;
    let mut start_reflect = 2usize;
    let mut shot: Option<String> = None;
    let mut check_camera = false;

    for arg in std::env::args().skip(1) {
        if let Some(value) = arg.strip_prefix("--backend=") {
            backend = parse_backend(value);
        } else if let Some(value) = arg.strip_prefix("--res=") {
            let wanted = value.to_ascii_uppercase();
            if let Some(i) = RESOLUTIONS.iter().position(|(name, _, _)| *name == wanted) {
                start_res = i;
            }
        } else if let Some(value) = arg.strip_prefix("--shot=") {
            shot = Some(value.to_string());
        } else if let Some(value) = arg.strip_prefix("--shadows=") {
            start_shadow = value.parse::<usize>().unwrap_or(2).min(3);
        } else if let Some(value) = arg.strip_prefix("--reflections=") {
            start_reflect = value.parse::<usize>().unwrap_or(2).min(3);
        } else if arg == "--check-camera" {
            check_camera = true;
        } else if arg == "--help" {
            println!(
                "reconl-tester [--backend=auto|vulkan|d3d11|soft-cpu] [--res=2K|1080P|920P|720P]\n\
                 \n\
                 WASD    move        SPACE/SHIFT  up / down\n\
                 ARROWS  look        MOUSE DRAG   look\n\
                 ESC     settings menu\n\
                 Tab     toggle statistics\n\
                 \n\
                 --shot=PATH        render one frame headlessly and write a PNG\n\
                 --shadows=0..3     start on a shadow level\n\
                 --reflections=0..3 start on a reflection level\n\
                 --check-camera     print the camera matrix against the reference pose"
            );
            return Ok(());
        }
    }

    if check_camera {
        // The tester's camera must agree with the host's fixed camera at the same
        // pose, or the scene is being viewed through a matrix nothing else uses.
        let ours = Camera::reference().view();
        let theirs = reconl_host::scene::CameraSpec {
            fov_y_deg: 55.0,
            near: 0.5,
            far: 120.0,
            eye_height: 12.0,
            eye_distance: 13.0,
        }
        .view();
        let mut worst = 0.0f32;
        for (a, b) in ours.iter().zip(theirs.iter()) {
            worst = worst.max((a - b).abs());
        }
        println!("worst element difference: {worst:.6}");
        println!("ours:   {:.4?}", ours);
        println!("theirs: {:.4?}", theirs);
        return Ok(());
    }

    let device = Device::create(&Config {
        backend,
        // The ladder would step the tier down behind the tester's back, which
        // would make every number here describe a device nobody asked for.
        target_frame_ms: 0,
        ..Config::default()
    })
    .map_err(|e| format!("creating the device: {e}"))?;

    let world = scene::build();
    let mut state = State {
        shadow: start_shadow,
        reflection: start_reflect,
        resolution: start_res,
        show_stats: true,
    };
    let mut camera = Camera::opening();
    let mut gfx = Gfx::new(
        &device,
        &world.chunks,
        RESOLUTIONS[state.resolution].1,
        RESOLUTIONS[state.resolution].2,
        256,
    )?;
    let mut pixels = vec![0u8; gfx.frame_bytes()];

    let scene = Scene {
        chunks: world.chunks.clone(),
        camera: world.camera,
        light: world.light,
        shadows: shadow_spec(state.shadow),
    };
    let mut scene = scene;
    let lights = [scene::sun(), scene::fill()];

    if let Some(path) = shot.as_deref() {
        let draws = build_draws(&world, state.reflection);
        let view = camera.view_for(gfx.width, gfx.height, 55.0, 0.5, 120.0);
        gfx.frame(&scene, &view, &lights, &draws, &mut pixels)?;
        let png = reconl_png::write_rgba8(gfx.width, gfx.height, &pixels)?;
        std::fs::write(path, png).map_err(|e| format!("{path}: {e}"))?;
        println!("{}x{} {} draws -> {path}", gfx.width, gfx.height, draws.len());
        println!("shadow map: {}", shadow_info(&device));
        return Ok(());
    }

    let mut buffer = vec![0u32; WINDOW_W * WINDOW_H];
    let mut overlay = vec![0u8; WINDOW_W * WINDOW_H * 4];

    let mut window = Window::new(
        "ReconL Tester",
        WINDOW_W,
        WINDOW_H,
        WindowOptions { resize: false, scale: Scale::X1, ..WindowOptions::default() },
    )
    .map_err(|e| format!("opening the window: {e}"))?;
    window.set_target_fps(0);

    let mut stats = Stats::new(240);
    let mut draws = build_draws(&world, state.reflection);
    let mut menu = Menu { open: false, index: 0 };
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut shadow_map = String::new();
    // The stats describe the last frame, so the readout is taken a frame after the
    // change rather than the frame of it.
    let mut shadow_dirty = 2u32;
    let mut last_frame = Instant::now();

    while window.is_open() {
        let dt = last_frame.elapsed().as_secs_f32().min(0.1);
        last_frame = Instant::now();

        // ---- menu --------------------------------------------------------
        let mut changes = Changes::default();
        if window.is_key_pressed(Key::Escape, KeyRepeat::No) {
            menu.open = !menu.open;
        }
        if menu.open {
            if window.is_key_pressed(Key::Up, KeyRepeat::Yes) {
                menu.index = (menu.index + MENU_ITEMS.len() - 1) % MENU_ITEMS.len();
            }
            if window.is_key_pressed(Key::Down, KeyRepeat::Yes) {
                menu.index = (menu.index + 1) % MENU_ITEMS.len();
            }
            if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
                changes = menu.adjust(&mut state, -1);
            }
            if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
                changes = menu.adjust(&mut state, 1);
            }
            if window.is_key_pressed(Key::Enter, KeyRepeat::No) && menu.index == MENU_ITEMS.len() - 1 {
                changes.quit = true;
            }
            if window.is_key_pressed(Key::Tab, KeyRepeat::No) {
                state.show_stats = !state.show_stats;
            }
        } else {
            // ---- movement ------------------------------------------------
            let speed = 9.0 * dt;
            let forward = camera.forward();
            let right = camera.right();
            // Movement follows the ground: walking, not flying at the horizon.
            let flat = normalize([forward[0], 0.0, forward[2]]);
            let strafe = normalize([right[0], 0.0, right[2]]);
            if window.is_key_down(Key::W) {
                camera.move_by(flat, speed);
            }
            if window.is_key_down(Key::S) {
                camera.move_by(flat, -speed);
            }
            if window.is_key_down(Key::D) {
                camera.move_by(strafe, speed);
            }
            if window.is_key_down(Key::A) {
                camera.move_by(strafe, -speed);
            }
            if window.is_key_down(Key::Space) {
                camera.move_by([0.0, 1.0, 0.0], speed);
            }
            if window.is_key_down(Key::LeftShift) {
                camera.move_by([0.0, 1.0, 0.0], -speed);
            }
            if window.is_key_down(Key::Up) {
                camera.look(0.0, 1.1 * dt);
            }
            if window.is_key_down(Key::Down) {
                camera.look(0.0, -1.1 * dt);
            }
            if window.is_key_down(Key::Left) {
                camera.look(-1.4 * dt, 0.0);
            }
            if window.is_key_down(Key::Right) {
                camera.look(1.4 * dt, 0.0);
            }

            // Mouse drag to look. The cursor is not captured - a tester that traps
            // the pointer is a tester you cannot alt-tab out of.
            let mouse = window.get_mouse_pos(MouseMode::Discard);
            if window.get_mouse_down(MouseButton::Left) {
                if let (Some((mx, my)), Some((lx, ly))) = (mouse, last_mouse) {
                    camera.look((mx - lx) * 0.004, -(my - ly) * 0.004);
                }
                last_mouse = mouse;
            } else {
                last_mouse = None;
            }

            // ---- direct shortcuts ---------------------------------------
            for level in 0..4usize {
                let key = [Key::Key1, Key::Key2, Key::Key3, Key::Key4][level];
                if window.is_key_pressed(key, KeyRepeat::No) && state.shadow != level {
                    state.shadow = level;
                    changes.shadow = true;
                }
            }
            if window.is_key_pressed(Key::Q, KeyRepeat::No) && state.reflection > 0 {
                state.reflection -= 1;
                changes.reflection = true;
            }
            if window.is_key_pressed(Key::E, KeyRepeat::No) && state.reflection < 3 {
                state.reflection += 1;
                changes.reflection = true;
            }
            if window.is_key_pressed(Key::R, KeyRepeat::No) {
                state.resolution = (state.resolution + 1) % RESOLUTIONS.len();
                changes.resolution = true;
            }
            if window.is_key_pressed(Key::Tab, KeyRepeat::No) {
                state.show_stats = !state.show_stats;
            }
        }

        if changes.quit {
            break;
        }
        if changes.shadow {
            scene.shadows = shadow_spec(state.shadow);
            shadow_dirty = 2;
            stats.clear();
        }
        if changes.reflection {
            draws = build_draws(&world, state.reflection);
            stats.clear();
        }
        if changes.resolution {
            let (_, w, h) = RESOLUTIONS[state.resolution];
            // A new resolution is a new swapchain, so the whole context is rebuilt
            // rather than resized: the depth buffer and the command list are sized
            // for the frame they serve.
            gfx = Gfx::new(&device, &world.chunks, w, h, 256)?;
            pixels = vec![0u8; gfx.frame_bytes()];
            stats.clear();
        }

        // ---- render ------------------------------------------------------
        let view = camera.view_for(gfx.width, gfx.height, 55.0, 0.5, 120.0);
        let ms = gfx.frame(&scene, &view, &lights, &draws, &mut pixels)?;
        // A frame spent inside the menu is a frame that did not render the scene
        // the statistics describe, so it is not recorded.
        if !menu.open {
            stats.push(ms);
        }
        if shadow_dirty > 0 {
            shadow_dirty -= 1;
            if shadow_dirty == 0 {
                shadow_map = shadow_info(&device);
            }
        }

        // Scale the rendered frame into the fixed-size window.
        let (fw, fh) = (gfx.width as usize, gfx.height as usize);
        for y in 0..WINDOW_H {
            let sy = y * fh / WINDOW_H;
            for x in 0..WINDOW_W {
                let sx = x * fw / WINDOW_W;
                let at = (sy * fw + sx) * 4;
                let r = pixels[at] as u32;
                let g = pixels[at + 1] as u32;
                let b = pixels[at + 2] as u32;
                buffer[y * WINDOW_W + x] = (r << 16) | (g << 8) | b;
            }
        }

        overlay.fill(0);
        draw_hud(&mut overlay, WINDOW_W as u32, WINDOW_H as u32, &state, &stats, &shadow_map);
        if menu.open {
            draw_menu(&mut overlay, WINDOW_W as u32, WINDOW_H as u32, &state, &menu);
        }
        composite(&mut buffer, &overlay);

        window
            .update_with_buffer(&buffer, WINDOW_W, WINDOW_H)
            .map_err(|e| format!("presenting the window: {e}"))?;
    }
    Ok(())
}

/// Copies the overlay's written pixels over the scene.
///
/// The overlay's alpha is composited into its own colour channels when it is
/// drawn, so a written pixel replaces the scene pixel and an untouched one leaves
/// it alone. Only the overlay is scanned: a full-frame pass here would add
/// milliseconds to every frame, and a tester that slows down the thing it is
/// timing reports its own cost as the renderer's.
fn composite(buffer: &mut [u32], overlay: &[u8]) {
    for i in 0..WINDOW_W * WINDOW_H {
        let at = i * 4;
        if overlay[at + 3] == 0 {
            continue;
        }
        buffer[i] = ((overlay[at] as u32) << 16) | ((overlay[at + 1] as u32) << 8) | overlay[at + 2] as u32;
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("reconl-tester: {e}");
        std::process::exit(1);
    }
}
