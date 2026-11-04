// Pixel shader for color pass
#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_world_pos;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_color;
layout(location = 4) in float v_textured;
layout(location = 5) in float v_lit;
layout(location = 6) in float v_receives_shadow;

layout(location = 0) uniform sampler2D white_tex;
layout(location = 1) uniform sampler2D shadow_map;
layout(location = 2) uniform sampler2D shadow_map_cascade; // Not used in simple case

// Push constants for lights (updated per draw)
layout(push_constant) uniform LightConstants {
    vec4 light_dir;
    float light_hash;
    float unused[2];
} lc;

layout(location = 0) out vec4 out_color;

// Shadow filter taps (PCF 3x3)
// The reference uses these exact tap offsets
vec2 tap_offsets[9] = vec2[](
    vec2(-1.0, -1.0), vec2(0.0, -1.0), vec2(1.0, -1.0),
    vec2(-1.0,  0.0), vec2(0.0,  0.0), vec2(1.0,  0.0),
    vec2(-1.0,  1.0), vec2(0.0,  1.0), vec2(1.0,  1.0)
);

float shadow_lookup(vec2 shadow_coord, float cascade_billboard) {
    // Perspective divide
    vec3 shadow_pos = shadow_coord.xyz / shadow_coord.w;
    if (shadow_pos.z < -1.0 || shadow_pos.z > 1.0 || shadow_pos.x < -1.0 || shadow_pos.x > 1.0 || shadow_pos.y < -1.0 || shadow_pos.y > 1.0) {
        return 1.0; // Outside shadow map - no shadow
    }

    // Convert to [0,1] range
    vec2 uv = shadow_pos.xy * 0.5 + 0.5;
    float depth = shadow_pos.z * 0.5 + 0.5;

    // PCF 3x3
    float shadow = 0.0;
    float texel_size = 1.0 / 512.0; // Will be updated based on actual map size
    for (int i = 0; i < 9; i++) {
        float d = textureOffset(shadow_map, uv, ivec2(tap_offsets[i] * texel_size * 512.0)).r;
        shadow += depth > d ? 1.0 : 0.0;
    }
    return shadow / 9.0;
}

// Simplified lighting model matching the reference
vec3 compute_lighting(vec3 world_pos, vec3 normal, vec3 view_pos, vec4 base_color, float lit) {
    if (lit < 0.5) {
        return base_color.rgb * 0.5; // Unlit ambient
    }

    vec3 light_direction = normalize(-lc.light_dir.xyz);
    float ndotl = max(dot(normal, light_direction), 0.0);

    // Ambient + diffuse
    vec3 ambient = base_color.rgb * 0.3;
    vec3 diffuse = base_color.rgb * ndotl * lc.light_dir.w; // light_dir.w is intensity

    return ambient + diffuse;
}

void main() {
    vec4 color = v_color;

    // Apply texture if textured
    if (v_textured > 0.5) {
        vec4 tex_color = texture(white_tex, v_uv);
        color.rgb *= tex_color.rgb;
    }

    // Apply lighting
    if (v_lit > 0.5) {
        vec3 lit_color = compute_lighting(v_world_pos, v_normal, v_world_pos, color, 1.0);
        color.rgb = lit_color;
    }

    // Apply shadow
    if (v_receives_shadow > 0.5) {
        // Shadow lookup would go here - simplified for now
        // In reality, this needs the shadow coordinate from the vertex shader
        float shadow = shadow_lookup(vec4(0.0), 0.0);
        color.rgb *= shadow;
    }

    // Blend mode
    float blend_mode = pc.flags[3];
    if (blend_mode > 0.5) {
        // Alpha blend: src_alpha, inv_src_alpha
        float alpha = color.a;
        out_color = vec4(color.rgb * alpha + vec3(0.0) * (1.0 - alpha), alpha);
    } else {
        out_color = color;
    }
}
