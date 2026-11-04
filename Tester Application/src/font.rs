//! A 5x7 bitmap font and the two drawing helpers the panel needs.
//!
//! A font file would be a dependency and an asset; a tester that has to find a
//! TTF next to its binary is a tester that fails on someone else's machine. Every
//! glyph is five columns of seven bits, which is enough to read a settings panel.

/// One glyph: the character and its seven rows, bit 4 the leftmost column.
type Glyph = (u8, [u8; 7]);

const GLYPHS: &[Glyph] = &[
    (b'A', [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    (b'B', [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110]),
    (b'C', [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110]),
    (b'D', [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110]),
    (b'E', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111]),
    (b'F', [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000]),
    (b'G', [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111]),
    (b'H', [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
    (b'I', [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    (b'J', [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100]),
    (b'K', [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001]),
    (b'L', [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111]),
    (b'M', [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001]),
    (b'N', [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001]),
    (b'O', [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    (b'P', [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000]),
    (b'Q', [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101]),
    (b'R', [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001]),
    (b'S', [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110]),
    (b'T', [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100]),
    (b'U', [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110]),
    (b'V', [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100]),
    (b'W', [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001]),
    (b'X', [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001]),
    (b'Y', [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100]),
    (b'Z', [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111]),
    (b'0', [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110]),
    (b'1', [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]),
    (b'2', [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111]),
    (b'3', [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110]),
    (b'4', [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010]),
    (b'5', [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110]),
    (b'6', [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110]),
    (b'7', [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000]),
    (b'8', [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110]),
    (b'9', [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100]),
    (b':', [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000]),
    (b'.', [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110]),
    (b'-', [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000]),
    (b'/', [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000]),
    (b'%', [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011]),
    (b'(', [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010]),
    (b')', [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000]),
    (b'+', [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000]),
    (b'[', [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110]),
    (b']', [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110]),
    (b'>', [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000]),
];

fn glyph(c: u8) -> Option<&'static [u8; 7]> {
    GLYPHS
        .iter()
        .find(|(ch, _)| *ch == c.to_ascii_uppercase())
        .map(|(_, rows)| rows)
}

/// Fills an axis-aligned rectangle, clipped to the frame.
pub fn rect(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    colour: [u8; 4],
) {
    let x0 = x.max(0).min(width as i32);
    let y0 = y.max(0).min(height as i32);
    let x1 = (x + w).max(0).min(width as i32);
    let y1 = (y + h).max(0).min(height as i32);
    let alpha = colour[3] as u32;
    if alpha == 0 {
        return;
    }
    for py in y0..y1 {
        for px in x0..x1 {
            let at = ((py as u32 * width + px as u32) * 4) as usize;
            if alpha == 255 {
                pixels[at..at + 4].copy_from_slice(&colour);
            } else {
                // Blend over whatever the renderer put there, so the panel does
                // not hide the scene behind it.
                for c in 0..3 {
                    let dst = pixels[at + c] as u32;
                    let src = colour[c] as u32;
                    pixels[at + c] = ((src * alpha + dst * (255 - alpha)) / 255) as u8;
                }
                pixels[at + 3] = 255;
            }
        }
    }
}

/// Draws `text` at `(x, y)` scaled by `scale`, returning the width it occupied.
pub fn text(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    scale: i32,
    text: &str,
    colour: [u8; 4],
) -> i32 {
    let mut pen = x;
    for byte in text.bytes() {
        if byte == b' ' {
            pen += 4 * scale;
            continue;
        }
        if let Some(rows) = glyph(byte) {
            for (row_index, bits) in rows.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        rect(
                            pixels,
                            width,
                            height,
                            pen + col * scale,
                            y + row_index as i32 * scale,
                            scale,
                            scale,
                            colour,
                        );
                    }
                }
            }
        }
        pen += 6 * scale;
    }
    pen - x
}

/// The width `text` will occupy at `scale`.
pub fn measure(text: &str, scale: i32) -> i32 {
    let mut total = 0;
    for byte in text.bytes() {
        total += if byte == b' ' { 4 * scale } else { 6 * scale };
    }
    total
}
