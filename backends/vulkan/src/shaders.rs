//! GLSL shader sources for the Vulkan backend.
//!
//! Ports of the HLSL shaders in `backends/d3d11/src/shaders.rs`.
//! Compiled to SPIR-V at build time by `build.rs` using the Vulkan SDK's `glslc`.

/// Vertex shader for the colour pass.
///
/// `gl_Position.y` is negated: Vulkan's NDC has +Y down, D3D11's has +Y up, and
/// the reference rasteriser works in a y-down screen space. One negation aligns
/// the three.
pub const VS_GLSL: &str = r#"#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 color;

layout(push_constant) uniform DrawPush {
    mat4 transform;
    vec4 flags;    // x textured, y lit, z receives_shadow, w blend mode
    vec4 flags1;   // x two_sided
} pc;

layout(location = 0) out vec3 v_world;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec4 v_color;

void main() {
    vec4 clip = pc.transform * vec4(position, 1.0);
    clip.y = -clip.y;
    gl_Position = clip;
    v_world = position;
    v_normal = normal;
    v_uv = uv;
    v_color = color;
}
"#;

/// Pixel shader for a depth-only pass.
///
/// The shadow pass has no colour attachment, but its pipeline shares the colour
/// pipelines' layout, whose push-constant range covers the fragment stage - and a
/// pipeline may only use a subset of its layout's stages. A pipeline with no
/// fragment stage at all broke that rule and drew nothing, silently: the map kept
/// its clear value and every tap compared against 0.0, so the frame came out fully
/// lit. A trivial fragment shader keeps the layout legal and writes nothing.
pub const PS_DEPTH_GLSL: &str = r#"#version 450
void main() {}
"#;

/// Pixel shader for the colour pass: the full lighting + shadow + blend path.
///
/// Bindings:
///   0 = base texture (white in this ABI, which carries no texture payload)
///   1 = shadow comparison sampler
///   2 = shadow raw depth sampler
///   3 = lights UBO
///   4 = shadow UBO
pub const PS_GLSL: &str = r#"#version 450

layout(location = 0) in vec3 v_world;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_color;

layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2D base_tex;
layout(binding = 1) uniform sampler2DArrayShadow shadow_cmp;
layout(binding = 2) uniform sampler2DArray shadow_raw;

layout(binding = 3) uniform LightsBlock {
    vec4 light_a[16];   // xyz position, w kind (0 dir, 1 spot, 2 point)
    vec4 light_b[16];   // xyz direction, w intensity
    vec4 light_c[16];   // xyz colour, w range
    vec4 light_d[16];   // x cos_inner, y cos_outer, z cast_shadow, w unused
    vec4 ambient_count; // xyz ambient, w light count
} L;

layout(binding = 4) uniform ShadowBlock {
    vec4 view_z_row;
    vec4 params_a;          // x filter_mode, y cascade_count, z normal_bias, w depth_bias
    vec4 params_b;          // x slope_bias, y max_distance, z blend_band, w map_size
    vec4 params_c;          // x map_width, y map_height
    mat4 cascade_mvp[4];
    vec4 cascade_params[4]; // x split_distance, y texel_world, z depth_span, w map_index
} S;

layout(push_constant) uniform DrawPush {
    mat4 transform;
    vec4 flags;
    vec4 flags1;
} pc;

// The reference `slope_bias_term`: tan(theta) capped at 8.0.
float slope_term(vec3 n, vec3 l) {
    float ndl = dot(n, l);
    if (ndl <= 1.0e-4) { return 8.0; }
    return min(sqrt(max(0.0, 1.0 - ndl * ndl)) / ndl, 8.0);
}

// One comparison tap: lit when reference + epsilon >= stored. Reversed-Z.
//
// The compare sampler filters LINEAR with COMPARE_OP enabled, so a single tap
// is a bilinear blend of four texel comparisons - hardware PCF for free. Every
// filter below is built on that, which is why four taps already beat the nine
// nearest taps this used to spend.
float tap(vec2 uv, float slice, float reference) {
    // A comparison sample of one cascade's layer: CLAMP_TO_EDGE plus the
    // sampler's COMPARE_OP, with the layer named by the cascade's map_index.
    return texture(shadow_cmp, vec4(uv, slice, reference + 1.0e-7));
}

// Interleaved gradient noise in [0, 1): the per-pixel kernel rotation that
// stops a four-tap kernel from showing its own square. A function of the pixel,
// never of time, so a static scene keeps a static image - no shimmer, and no
// temporal history to leak a frame through.
float kernel_rotation(vec2 frag) {
    return fract(52.9829189 * fract(dot(frag, vec2(0.06711056, 0.00583715)))) * 6.2831853;
}

// Eight bilinear taps on a rotated ring: 32 texel comparisons over the same
// `2 * radius + 2` footprint. The wider kernel a 5x5 request buys, spent on
// samples rather than on a bigger box, so the penumbra is a ramp and not a
// dither of four.
const vec2 RING8[8] = vec2[8](
    vec2( 1.0000000,  0.0000000), vec2( 0.7071068,  0.7071068),
    vec2( 0.0000000,  1.0000000), vec2(-0.7071068,  0.7071068),
    vec2(-1.0000000,  0.0000000), vec2(-0.7071068, -0.7071068),
    vec2( 0.0000000, -1.0000000), vec2( 0.7071068, -0.7071068)
);

float pcf8(vec2 uv, float slice, float reference, float radius) {
    vec2 texel = 1.0 / vec2(S.params_c.x, S.params_c.y);
    float a = kernel_rotation(gl_FragCoord.xy);
    mat2 rot = mat2(cos(a), sin(a), -sin(a), cos(a));
    vec2 o = texel * radius;
    float sum = 0.0;
    for (int i = 0; i < 8; ++i) {
        sum += tap(uv + (rot * RING8[i]) * o, slice, reference);
    }
    return sum * 0.125;
}

// pcss-lite: raw-depth 3x3 blocker search, then a 5x5 comparison grid.
float pcss_lite(vec2 uv, float slice, float reference) {
    vec2 texel = 1.0 / vec2(S.params_c.x, S.params_c.y);
    float blocker_sum = 0.0;
    float blockers = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float d = textureLod(shadow_raw,
                vec3(clamp(uv + vec2(dx, dy) * texel, 0.0, 1.0), slice), 0.0).r;
            if (d > reference) {
                blocker_sum += d;
                blockers += 1.0;
            }
        }
    }
    if (blockers == 0.0) { return 1.0; }
    float avg = blocker_sum / blockers;
    float width = clamp(16.0 * max(avg - reference, 0.0) / max(avg, 1.0e-4), 0.0, 4.0);
    float sum = 0.0;
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            sum += tap(uv + vec2(i, j) * (width * 0.5) * texel, slice, reference);
        }
    }
    return sum / 25.0;
}

float lookup_at(int ci, vec3 world, vec3 n, vec3 l) {
    vec4 cp = S.cascade_params[ci];
    mat4 mvp = S.cascade_mvp[ci];

    float texel_world = max(cp.y, 1.0e-6);
    vec3 wp = world + n * (S.params_a.z * texel_world);
    vec4 lp = mvp * vec4(wp, 1.0);
    float w = lp.w;
    if (abs(w) < 1.0e-9) { return 1.0; }
    vec3 ndc = lp.xyz / w;
    vec2 uv = vec2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }

    float slope = slope_term(n, l);
    float depth_span = max(abs(cp.z), 1.0e-6);
    float bias = S.params_a.w + S.params_b.x * texel_world * slope / depth_span;
    float reference = ndc.z + bias;

    // The kernel radius is a *world* size divided by this cascade's texel
    // size, so the penumbra stays the same width on the ground as the cascade
    // changes instead of doubling when the map does. Clamped so a near cascade
    // cannot turn one tap into a blur and a far one cannot lose it entirely.
    float penumbra = clamp(S.params_c.z / texel_world, 1.0, 2.0);

    float filter_mode = S.params_a.x;
    if (filter_mode < 0.5) { return tap(uv, cp.w, reference); }
    // Eight rotated bilinear taps is where the crosshatch of a four-tap
    // kernel stops being visible on a 2K map, and it is still fewer texel
    // fetches than the nine nearest taps it replaced.
    if (filter_mode < 1.5) { return pcf8(uv, cp.w, reference, penumbra); }
    if (filter_mode < 2.5) { return pcf8(uv, cp.w, reference, penumbra + 1.0); }
    return pcss_lite(uv, cp.w, reference);
}

float shadow_factor(vec3 world, vec3 n, vec3 l) {
    int count = int(S.params_a.y);
    float max_distance = S.params_b.y;
    float blend_band = S.params_b.z;

    float distance = -(S.view_z_row.x * world.x + S.view_z_row.y * world.y
                     + S.view_z_row.z * world.z + S.view_z_row.w);
    if (distance > max_distance || count <= 0) { return 1.0; }

    int ci = count - 1;
    for (int k = 0; k < 3; ++k) {
        if (k < count && distance <= S.cascade_params[k].x) { ci = k; break; }
    }
    float lit = lookup_at(ci, world, n, l);

    if (blend_band > 0.0 && distance > S.cascade_params[ci].x - blend_band) {
        float t = clamp((distance - (S.cascade_params[ci].x - blend_band)) / blend_band, 0.0, 1.0);
        if (ci + 1 < count) {
            float f = lookup_at(ci + 1, world, n, l);
            lit = lit + (f - lit) * t;
        } else {
            lit = lit + (1.0 - lit) * t;
        }
    }
    if (blend_band > 0.0 && distance > max_distance - blend_band) {
        float t = clamp((distance - (max_distance - blend_band)) / blend_band, 0.0, 1.0);
        lit = lit + (1.0 - lit) * t;
    }
    return lit;
}

void main() {
    bool textured = pc.flags.x > 0.5;
    bool lit = pc.flags.y > 0.5;
    bool receives_shadow = pc.flags.z > 0.5;

    vec4 texel = vec4(1.0);
    if (textured) { texel = texture(base_tex, v_uv); }

    // two_sided: the reference flips the normal on back faces before normalise.
    vec3 nrm = v_normal;
    if (pc.flags1.x > 0.5 && !gl_FrontFacing) { nrm = -nrm; }
    vec3 n = normalize(nrm);
    vec3 albedo = v_color.rgb * texel.rgb;
    float a = clamp(v_color.a * texel.a, 0.0, 1.0);

    vec3 rgb;
    if (!lit) {
        rgb = clamp(albedo, 0.0, 1.0);
    } else {
        vec3 acc = L.ambient_count.xyz;
        int count = int(L.ambient_count.w);
        for (int i = 0; i < 16; ++i) {
            if (i >= count) { break; }
            vec3 to_light;
            float attenuation;
            if (L.light_a[i].w < 0.5) {
                to_light = normalize(-L.light_b[i].xyz);
                attenuation = 1.0;
            } else if (L.light_a[i].w > 1.5) {
                vec3 delta = L.light_a[i].xyz - v_world;
                float d = length(delta);
                to_light = (d > 1.0e-6) ? delta / d : vec3(0.0, 1.0, 0.0);
                float fa = (L.light_c[i].w <= 0.0) ? 1.0
                    : clamp(1.0 - d / L.light_c[i].w, 0.0, 1.0);
                attenuation = fa * fa;
            } else {
                vec3 delta = L.light_a[i].xyz - v_world;
                float d = length(delta);
                to_light = (d > 1.0e-6) ? delta / d : vec3(0.0, 1.0, 0.0);
                float cos_theta = -dot(normalize(L.light_b[i].xyz), to_light);
                float inner = L.light_d[i].x;
                float outer = L.light_d[i].y;
                float spot;
                if (inner - outer <= 1.0e-6) {
                    spot = (cos_theta >= outer) ? 1.0 : 0.0;
                } else {
                    spot = clamp((cos_theta - outer) / (inner - outer), 0.0, 1.0);
                }
                float fa = (L.light_c[i].w <= 0.0) ? 1.0
                    : clamp(1.0 - d / L.light_c[i].w, 0.0, 1.0);
                attenuation = fa * fa * spot;
            }
            float ndl = max(dot(n, to_light), 0.0);
            if (ndl <= 0.0) { continue; }
            float shadow = 1.0;
            if (receives_shadow && L.light_d[i].z > 0.5) {
                shadow = shadow_factor(v_world, n, to_light);
            }
            float k = L.light_b[i].w * ndl * attenuation * shadow;
            acc = acc + L.light_c[i].xyz * k;
        }
        rgb = clamp(albedo * acc, 0.0, 1.0);
    }

    // Blend pre-multiplication matches the d3d11 backend's fixed-function states.
    int blend_mode = int(pc.flags.w + 0.5);
    if (blend_mode == 0) {
        out_color = vec4(rgb, 1.0);
    } else if (blend_mode == 2 || blend_mode == 3) {
        out_color = vec4(rgb * a, a);
    } else {
        out_color = vec4(rgb, a);
    }
}
"#;

/// Vertex shader for the depth (shadow) pass.
pub const VS_DEPTH_GLSL: &str = r#"#version 450

layout(location = 0) in vec3 position;

layout(push_constant) uniform DrawPush {
    mat4 transform;
    vec4 flags;
    vec4 flags1;
} pc;

void main() {
    vec4 clip = pc.transform * vec4(position, 1.0);
    clip.y = -clip.y;   // Vulkan NDC is +Y down; see VS_GLSL.
    gl_Position = clip;
}
"#;
