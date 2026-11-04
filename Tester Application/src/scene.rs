//! The test world: a walled courtyard with walls, pillars, and squares, circles
//! and triangles arranged around it.
//!
//! Geometry is built once in local space; the world is a list of draws that place
//! those chunks with a model matrix. That keeps the vertex buffers static - the
//! renderer uploads them at creation and never again - while the scene still has
//! many more objects than it has meshes.

use crate::gfx::{Draw, Pass};
use reconl_host::scene::{CameraSpec, Chunk, LightSpec, V};

/// Chunk indices. Fixed, because the renderer builds one vertex buffer per chunk
/// at creation and the draw list may only name chunks that exist.
pub const FLOOR: usize = 0;
pub const FLOOR_GLASS: usize = 1;
pub const BOX: usize = 2;
pub const DISC: usize = 3;
pub const TRI_FLAT: usize = 4;
pub const TRI_UP: usize = 5;
pub const SQUARE: usize = 6;
/// Reflected copies of `BOX..=SQUARE` start here: each is the same mesh with its
/// normals' Y negated, because the mirror matrix negates it again at draw time.
pub const REFLECT_BASE: usize = 7;

/// Object classes, used to decide what a reflection level includes.
pub const TAG_FLOOR: u8 = 0;
pub const TAG_WALL: u8 = 1;
pub const TAG_PILLAR: u8 = 2;
pub const TAG_DISC: u8 = 3;
pub const TAG_SHAPE: u8 = 4;

pub struct World {
    pub chunks: Vec<Chunk>,
    pub draws: Vec<Draw>,
    pub camera: CameraSpec,
    pub light: LightSpec,
}

fn v(pos: [f32; 3], nrm: [f32; 3], col: [f32; 4]) -> V {
    V { pos, nrm, uv: [0.0, 0.0], col }
}

/// A unit square in the XZ plane, normal +Y, centred on the origin.
fn plane(col: [f32; 4]) -> Chunk {
    let up = [0.0, 1.0, 0.0];
    Chunk {
        verts: vec![
            v([-0.5, 0.0, -0.5], up, col),
            v([0.5, 0.0, -0.5], up, col),
            v([0.5, 0.0, 0.5], up, col),
            v([-0.5, 0.0, 0.5], up, col),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

/// A unit cube centred on the origin, one colour, flat normals per face.
fn cube(col: [f32; 4]) -> Chunk {
    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    let h = 0.5;
    // (normal, four corners in counter-clockwise order seen from outside)
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]]),
        ([0.0, 0.0, -1.0], [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]]),
        ([1.0, 0.0, 0.0], [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]]),
        ([-1.0, 0.0, 0.0], [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]]),
        ([0.0, 1.0, 0.0], [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]]),
        ([0.0, -1.0, 0.0], [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]]),
    ];
    for (normal, corners) in faces {
        let base = verts.len() as u32;
        for corner in corners {
            verts.push(v(corner, normal, col));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Chunk { verts, indices }
}

/// A unit-diameter disc in the XZ plane, normal +Y.
fn disc(segments: usize, col: [f32; 4]) -> Chunk {
    let up = [0.0, 1.0, 0.0];
    let mut verts = vec![v([0.0, 0.0, 0.0], up, col)];
    let mut indices = Vec::new();
    for i in 0..segments {
        let a = (i as f32) * std::f32::consts::TAU / segments as f32;
        verts.push(v([a.cos() * 0.5, 0.0, a.sin() * 0.5], up, col));
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        indices.extend_from_slice(&[0, i as u32 + 1, next as u32 + 1]);
    }
    Chunk { verts, indices }
}

/// A unit triangle lying flat in the XZ plane, normal +Y.
fn tri_flat(col: [f32; 4]) -> Chunk {
    let up = [0.0, 1.0, 0.0];
    Chunk {
        verts: vec![
            v([0.0, 0.0, -0.5], up, col),
            v([0.5, 0.0, 0.5], up, col),
            v([-0.5, 0.0, 0.5], up, col),
        ],
        indices: vec![0, 1, 2],
    }
}

/// A unit triangle standing upright in the XY plane, normal +Z.
fn tri_up(col: [f32; 4]) -> Chunk {
    let n = [0.0, 0.0, 1.0];
    Chunk {
        verts: vec![
            v([0.0, 1.0, 0.0], n, col),
            v([-0.5, 0.0, 0.0], n, col),
            v([0.5, 0.0, 0.0], n, col),
        ],
        indices: vec![0, 1, 2],
    }
}

/// A mesh prepared for being drawn through `MIRROR_Y`.
///
/// The mirror matrix has a negative determinant, so it flips a vertex normal's Y
/// as surely as it flips a position - which lights the reflection from below and
/// turns it black. Negating Y in the mesh cancels that at draw time, so a
/// reflection is shaded exactly like the object it reflects, and the translucent
/// floor is then the only thing dimming it.
fn mirrored(chunk: &Chunk) -> Chunk {
    let mut verts = chunk.verts.clone();
    for v in &mut verts {
        v.nrm[1] = -v.nrm[1];
    }
    Chunk { verts, indices: chunk.indices.clone() }
}

/// A translate-and-scale model matrix, column-major.
pub fn trs(scale: [f32; 3], pos: [f32; 3]) -> [f32; 16] {
    [
        scale[0], 0.0, 0.0, 0.0, //
        0.0, scale[1], 0.0, 0.0, //
        0.0, 0.0, scale[2], 0.0, //
        pos[0], pos[1], pos[2], 1.0,
    ]
}

/// A rotation about +Y, column-major.
pub fn rot_y(radians: f32) -> [f32; 16] {
    let (s, c) = radians.sin_cos();
    [
        c, 0.0, -s, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        s, 0.0, c, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Mirror about the ground plane: the matrix that turns a reflection into a draw.
pub const MIRROR_Y: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, -1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

pub fn build() -> World {
    // Mid tones, not near-black. A shadow can only read as a shadow if the lit
    // surface is bright enough to fall from: at 0.10 grey a lit floor came out at
    // 18/255 and its shadow at 1/255, which reads as a hole rather than as shade.
    // The ambient term is 6% and is not settable through the ABI, so the albedo
    // is what a scene controls.
    let slate = [0.50, 0.52, 0.55, 1.0];
    let mut glass = slate;
    // The floor is drawn over the mirrored copies, so it has to let them through.
    glass[3] = 0.70;
    let stone = [0.70, 0.71, 0.74, 1.0];
    let amber = [0.96, 0.56, 0.12, 1.0];
    let cyan = [0.12, 0.68, 0.86, 1.0];
    let magenta = [0.82, 0.22, 0.52, 1.0];
    // The squares need their own chunk: they were drawn with FLOOR, the same
    // chunk as the ground, so they were the same colour as the thing they sat on.
    let cream = [0.84, 0.78, 0.66, 1.0];

    let chunks = vec![
        plane(slate),
        plane(glass),
        cube(stone),
        disc(28, amber),
        tri_flat(cyan),
        tri_up(magenta),
        plane(cream),
    ];
    let mut chunks = chunks;
    for index in BOX..=SQUARE {
        let copy = mirrored(&chunks[index]);
        chunks.push(copy);
    }

    let mut draws = Vec::with_capacity(64);

    // Four walls around a 20-unit courtyard. Boxes, so they both receive and cast.
    for (scale, pos) in [
        ([20.0, 2.6, 0.5], [0.0, 1.3, -10.0]),
        ([20.0, 2.6, 0.5], [0.0, 1.3, 10.0]),
        ([0.5, 2.6, 20.0], [-10.0, 1.3, 0.0]),
        ([0.5, 2.6, 20.0], [10.0, 1.3, 0.0]),
    ] {
        draws.push(Draw {
            chunk: BOX,
            model: trs(scale, pos),
            pass: Pass::Opaque,
            tag: TAG_WALL,
        });
    }
    // Corner pillars: tall enough to throw a long shadow across the floor.
    for (x, z) in [(-6.5, -6.5), (6.5, -6.5), (-6.5, 6.5), (6.5, 6.5)] {
        draws.push(Draw {
            chunk: BOX,
            model: trs([0.9, 5.2, 0.9], [x, 2.6, z]),
            pass: Pass::Opaque,
            tag: TAG_PILLAR,
        });
    }

    // A ring of discs, hovering so each casts its own shadow.
    for i in 0..8 {
        let a = (i as f32) * std::f32::consts::TAU / 8.0;
        draws.push(Draw {
            chunk: DISC,
            model: trs([2.0, 1.0, 2.0], [a.cos() * 5.0, 1.6 + (i % 3) as f32 * 0.45, a.sin() * 5.0]),
            pass: Pass::Opaque,
            tag: TAG_DISC,
        });
    }

    // Flat triangles between the discs, at a lower height.
    for i in 0..6 {
        let a = (i as f32 + 0.5) * std::f32::consts::TAU / 6.0;
        draws.push(Draw {
            chunk: TRI_FLAT,
            model: trs([2.6, 1.0, 2.6], [a.cos() * 8.0, 0.5, a.sin() * 8.0]),
            pass: Pass::Opaque,
            tag: TAG_SHAPE,
        });
    }

    // Standing triangles, rotated to face the camera from either side.
    for (i, (x, z)) in [(-3.0, -2.0), (3.2, 2.4)].iter().enumerate() {
        let mut model = trs([2.4, 2.4, 2.4], [*x, 1.2, *z]);
        model = reconl_host::math::mul(&rot_y(0.35 + i as f32 * 0.6), &model);
        draws.push(Draw {
            chunk: TRI_UP,
            model,
            pass: Pass::Opaque,
            tag: TAG_SHAPE,
        });
    }

    // Squares laid flat, low to the ground.
    for (x, z, s) in [(-7.0, 2.5, 3.0), (7.0, -2.5, 3.0), (0.0, -6.8, 3.6)] {
        draws.push(Draw {
            chunk: SQUARE,
            model: trs([s, 1.0, s], [x, 0.35, z]),
            pass: Pass::Opaque,
            tag: TAG_SHAPE,
        });
    }

    World {
        chunks,
        draws,
        camera: CameraSpec {
            fov_y_deg: 55.0,
            near: 0.5,
            far: 120.0,
            eye_height: 12.0,
            eye_distance: 13.0,
        },
        light: sun(),
    }
}

/// The floor quad, drawn opaque when there are no reflections.
pub fn floor_opaque() -> Draw {
    Draw {
        chunk: FLOOR,
        model: trs([40.0, 1.0, 40.0], [0.0, 0.0, 0.0]),
        pass: Pass::Opaque,
        tag: TAG_FLOOR,
    }
}

/// The floor quad, drawn translucent over the mirrored copies.
pub fn floor_glass() -> Draw {
    Draw {
        chunk: FLOOR_GLASS,
        model: trs([40.0, 1.0, 40.0], [0.0, 0.0, 0.0]),
        pass: Pass::Alpha,
        tag: TAG_FLOOR,
    }
}

/// A shadowless fill light standing in for the sky.
///
/// The engine's ambient term is a fixed 6% and no ABI field sets it, so a scene
/// lit by one sun has shadows at 6% of its albedo - black. A dim second light
/// from the opposite side is how a scene without an ambient control gets light
/// into its shadows, and it is why the shadowed floor reads as blue-grey shade
/// rather than as a hole.
pub fn fill() -> LightSpec {
    LightSpec {
        direction: [-0.35, -0.72, -0.60],
        color: [0.60, 0.70, 1.0],
        intensity: 0.42,
        cast_shadow: false,
    }
}

/// The sun: about 35 degrees up, so a 5.2-unit pillar throws a 7.4-unit shadow -
/// long enough to read as a shadow, shallow enough that the courtyard is not
/// entirely in the walls' shade.
pub fn sun() -> LightSpec {
    LightSpec {
        direction: [0.35, -0.58, 0.74],
        color: [1.0, 0.96, 0.88],
        intensity: 1.45,
        cast_shadow: true,
    }
}
