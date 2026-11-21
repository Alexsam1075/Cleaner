// Fused real-time SDF ray marcher: one thread = one pixel, whole shading in registers.
// Scene = table of primitives (stride PS floats) + table of materials (stride MS floats).
#define PS 32
#define MS 24
#define PI 3.14159265f

struct V { float x, y, z; };
__device__ __forceinline__ V v3(float a, float b, float c) { V r; r.x = a; r.y = b; r.z = c; return r; }
__device__ __forceinline__ V operator+(V a, V b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ V operator-(V a, V b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ V operator*(V a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ V operator*(float s, V a) { return v3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ V mul(V a, V b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ V operator-(V a) { return v3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float len(V a) { return sqrtf(dot(a, a)); }
__device__ __forceinline__ V nrm(V a) { return a * rsqrtf(fmaxf(dot(a, a), 1e-20f)); }
__device__ __forceinline__ V vabs(V a) { return v3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }
__device__ __forceinline__ V vmax(V a, float b) { return v3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b)); }
__device__ __forceinline__ float clamp01(float x) { return fminf(fmaxf(x, 0.f), 1.f); }
__device__ __forceinline__ V lerp(V a, V b, float t) { return a + (b - a) * t; }
__device__ __forceinline__ V reflect(V d, V n) { return d - n * (2.f * dot(d, n)); }

// ------------------------------------------------------------------ noise
__device__ __forceinline__ float hash3(float x, float y, float z) {
  float h = x * 127.1f + y * 311.7f + z * 74.7f;
  float s = sinf(h) * 43758.5453f;
  return s - floorf(s);
}
__device__ float vnoise(V p) {
  float ix = floorf(p.x), iy = floorf(p.y), iz = floorf(p.z);
  float fx = p.x - ix, fy = p.y - iy, fz = p.z - iz;
  float ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy), uz = fz * fz * (3 - 2 * fz);
  float a = hash3(ix, iy, iz), b = hash3(ix + 1, iy, iz), c = hash3(ix, iy + 1, iz), d = hash3(ix + 1, iy + 1, iz);
  float e = hash3(ix, iy, iz + 1), f = hash3(ix + 1, iy, iz + 1), g = hash3(ix, iy + 1, iz + 1), h = hash3(ix + 1, iy + 1, iz + 1);
  float k0 = a + (b - a) * ux, k1 = c + (d - c) * ux, k2 = e + (f - e) * ux, k3 = g + (h - g) * ux;
  float l0 = k0 + (k1 - k0) * uy, l1 = k2 + (k3 - k2) * uy;
  return (l0 + (l1 - l0) * uz) * 2.f - 1.f;
}
__device__ float fbm(V p, int oct) {
  float amp = 1.f, tot = 0.f, nm = 0.f;
  for (int i = 0; i < oct; i++) { tot += amp * vnoise(p); nm += amp; p = p * 2.f + v3(17.3f, 17.3f, 17.3f); amp *= 0.5f; }
  return tot / nm;
}

// ------------------------------------------------------------------ primitives (local frame)
__device__ __forceinline__ float sd_box(V p, V b) {
  V q = vabs(p) - b;
  return len(vmax(q, 0.f)) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.f);
}
__device__ __forceinline__ float sd_rcyl(V p, float r, float h, float rr) {
  float dx = sqrtf(p.x * p.x + p.y * p.y) - (r - rr), dy = fabsf(p.z) - (h - rr);
  return sqrtf(fmaxf(dx, 0.f) * fmaxf(dx, 0.f) + fmaxf(dy, 0.f) * fmaxf(dy, 0.f)) + fminf(fmaxf(dx, dy), 0.f) - rr;
}
__device__ __forceinline__ float sd_capsule(V p, V a, V b, float r) {
  V pa = p - a, ba = b - a;
  float h = clamp01(dot(pa, ba) / dot(ba, ba));
  return len(pa - ba * h) - r;
}
__device__ __forceinline__ float sd_ellipsoid(V p, V r) {
  float k0 = len(v3(p.x / r.x, p.y / r.y, p.z / r.z));
  float k1 = len(v3(p.x / (r.x * r.x), p.y / (r.y * r.y), p.z / (r.z * r.z)));
  return k0 * (k0 - 1.f) / fmaxf(k1, 1e-9f);
}
__device__ __forceinline__ float sd_torus(V p, float R, float r) {
  float q = sqrtf(p.x * p.x + p.y * p.y) - R;
  return sqrtf(q * q + p.z * p.z) - r;
}
__device__ __forceinline__ float smin(float a, float b, float k) {
  float h = clamp01(0.5f + 0.5f * (b - a) / k);
  return b * (1 - h) + a * h - k * h * (1 - h);
}

// prim layout: 0 type,1 mat,2 obj,3 group,4-6 pos,7-15 rot(world->local rows),16-23 params,
//              24 disp_amp,25 disp_freq,26 disp_seed,27 smooth_k,28 bound,29 flags,30 shadowless
__device__ __forceinline__ V to_local(const float* P, V p) {
  V d = p - v3(P[4], P[5], P[6]);
  return v3(P[7] * d.x + P[8] * d.y + P[9] * d.z, P[10] * d.x + P[11] * d.y + P[12] * d.z,
            P[13] * d.x + P[14] * d.y + P[15] * d.z);
}
__device__ __forceinline__ V dir_to_world(const float* P, V v) {   // transpose of rot
  return v3(P[7] * v.x + P[10] * v.y + P[13] * v.z, P[8] * v.x + P[11] * v.y + P[14] * v.z,
            P[9] * v.x + P[12] * v.y + P[15] * v.z);
}
__device__ float prim_sdf(const float* P, V wp) {
  V q = to_local(P, wp);
  int type = (int)P[0];
  const float* a = P + 16;
  float d;
  if (type == 0) d = len(q) - a[0];
  else if (type == 1) d = sd_box(q, v3(a[0] - a[3], a[1] - a[3], a[2] - a[3])) - a[3];
  else if (type == 2) d = sd_rcyl(q, a[0], a[1], a[2]);
  else if (type == 3) d = sd_capsule(q, v3(a[0], a[1], a[2]), v3(a[3], a[4], a[5]), a[6]);
  else if (type == 4) d = sd_ellipsoid(q, v3(a[0], a[1], a[2]));
  else d = sd_torus(q, a[0], a[1]);
  int flags = (int)P[29];
  if (P[24] != 0.f) {
    float n = fbm(q * P[25] + v3(P[26], P[26], P[26]), (flags & 4) ? 3 : 4);
    if (flags & 1) n = fabsf(n) - 0.25f;
    d += P[24] * n;
  }
  if (flags & 2) d = fmaxf(d, -wp.z);             // flat where it rests on the floor
  return d;
}

struct Hit { float d; int prim; };
__device__ Hit scene(const float* prims, int n, V p, bool shadow_pass) {
  Hit best; best.d = 1e3f; best.prim = -1;
  float gd = 1e3f; int gprim = -1; float gid = -1e9f;
  for (int i = 0; i < n; i++) {
    const float* P = prims + i * PS;
    if (shadow_pass && P[30] > 0.5f) continue;
    float bd = len(p - v3(P[4], P[5], P[6])) - P[28];
    float d = bd > 0.02f ? bd + 0.01f : prim_sdf(P, p);
    if (P[3] >= 0.f && P[3] == gid) {
      float nd = P[27] > 0.f ? smin(gd, d, P[27]) : fminf(gd, d);
      if (d < gd) gprim = i;
      gd = nd;
    } else {
      if (gd < best.d) { best.d = gd; best.prim = gprim; }
      gd = d; gprim = i; gid = P[3];
    }
  }
  if (gd < best.d) { best.d = gd; best.prim = gprim; }
  return best;
}
__device__ float sentry(const float* prims, int n, V o, V d) {    // first bounding-sphere entry
  float best = 1e30f;
  for (int i = 0; i < n; i++) {
    const float* P = prims + i * PS;
    V oc = o - v3(P[4], P[5], P[6]);
    float b = dot(oc, d), c = dot(oc, oc) - P[28] * P[28], disc = b * b - c;
    if (disc > 0.f) {
      float s = sqrtf(disc), t0 = -b - s, t1 = -b + s;
      if (t1 > 0.f) best = fminf(best, fmaxf(t0, 0.f));
    }
  }
  return best;
}
__device__ bool march(const float* prims, int n, V o, V d, float tmax, int steps, float& t, int& prim, float sgn) {
  for (int i = 0; i < steps; i++) {
    Hit h = scene(prims, n, o + d * t, false);
    float dist = h.d * sgn;
    if (fabsf(dist) < 2e-4f + 1e-4f * t) { prim = h.prim; return true; }
    t += fmaxf(dist, 2e-4f) * 0.9f;
    if (t > tmax) return false;
  }
  return false;
}
__device__ V normal_at(const float* prims, int n, V p) {
  const float e = 2e-4f;
  V k0 = v3(1, -1, -1), k1 = v3(-1, -1, 1), k2 = v3(-1, 1, -1), k3 = v3(1, 1, 1);
  return nrm(k0 * scene(prims, n, p + k0 * e, false).d + k1 * scene(prims, n, p + k1 * e, false).d +
             k2 * scene(prims, n, p + k2 * e, false).d + k3 * scene(prims, n, p + k3 * e, false).d);
}
__device__ float soft_shadow(const float* prims, int n, V p, V l, float tmax, float k, float jit) {
  float res = 1.f, t = 0.004f + jit * 0.01f;
  for (int i = 0; i < 40 && t < tmax; i++) {
    float d = scene(prims, n, p + l * t, true).d;
    res = fminf(res, clamp01(k * d / t));
    if (res < 0.01f) break;
    t += fminf(fmaxf(d, 0.002f), 0.03f);
  }
  return powf(clamp01(res), 1.5f);
}
__device__ float ao_at(const float* prims, int n, V p, V nn, float scale) {
  float occ = 0.f;
  for (int i = 1; i <= 5; i++) {
    float h = scale * i / 5.f;
    occ += fmaxf(h - scene(prims, n, p + nn * h, true).d, 0.f) / (float)(1 << i);
  }
  return fminf(fmaxf(1.f - 3.f * occ / scale, 0.15f), 1.f);
}
__device__ V refract_dir(V d, V n, float eta) {
  float ci = -dot(d, n), k = 1.f - eta * eta * (1.f - ci * ci);
  if (k < 0.f) return reflect(d, n);
  return nrm(d * eta + n * (eta * ci - sqrtf(k)));
}

// ------------------------------------------------------------------ textures
struct Img { const float* p; int w, h, levels; };      // (levels, h, w, 3) float linear
__device__ V tex_fetch(Img im, int lvl, float x, float y) {   // bilinear, clamp
  x = fminf(fmaxf(x, 0.f), im.w - 1.001f); y = fminf(fmaxf(y, 0.f), im.h - 1.001f);
  int x0 = (int)x, y0 = (int)y; float fx = x - x0, fy = y - y0;
  const float* b = im.p + (size_t)lvl * im.w * im.h * 3;
  const float* a00 = b + (y0 * im.w + x0) * 3; const float* a01 = a00 + 3;
  const float* a10 = a00 + im.w * 3; const float* a11 = a10 + 3;
  V c00 = v3(a00[0], a00[1], a00[2]), c01 = v3(a01[0], a01[1], a01[2]);
  V c10 = v3(a10[0], a10[1], a10[2]), c11 = v3(a11[0], a11[1], a11[2]);
  return lerp(lerp(c00, c01, fx), lerp(c10, c11, fx), fy);
}
__device__ V floor_albedo(Img ft, float sx, float sy, float x, float y, float foot) {
  float u = fmodf(fabsf(x / sx), 2.f), v = fmodf(fabsf(y / sy), 2.f);
  if (u > 1.f) u = 2.f - u;
  if (v > 1.f) v = 2.f - v;
  float px = v * (ft.w - 1), py = u * (ft.h - 1);              // image x = floor y
  float lvl = fminf(fmaxf(log2f(fmaxf(foot / (sx / ft.h), 1.f)), 0.f), ft.levels - 1.001f);
  int l0 = (int)lvl;
  return lerp(tex_fetch(ft, l0, px, py), tex_fetch(ft, l0 + 1, px, py), lvl - l0);
}

// material layout: 0-2 color,3-5 color2,6 rough,7 metal,8 trans,9 ior,10-12 tint,13 tex,14 scale,15 amp,16 clearcoat
__device__ float star_dist(float y, float z, float r, float thick) {
  const float k1x = 0.809016994375f, k1y = -0.587785252292f;
  float px = fabsf(y), py = z;
  float d1 = fmaxf(k1x * px + k1y * py, 0.f); px -= 2 * d1 * k1x; py -= 2 * d1 * k1y;
  float d2 = fmaxf(-k1x * px + k1y * py, 0.f); px -= 2 * d2 * -k1x; py -= 2 * d2 * k1y;
  px = fabsf(px); py -= r;
  float bax = 0.42f * -k1y, bay = 0.42f * k1x - 1.f;
  float h = fminf(fmaxf((px * bax + py * bay) / (bax * bax + bay * bay), 0.f), r);
  float dd = sqrtf((px - bax * h) * (px - bax * h) + (py - bay * h) * (py - bay * h));
  float s = (py * bax - px * bay) > 0.f ? 1.f : -1.f;
  return fabsf(dd * s) - thick;
}
__device__ V albedo(const float* M, const float* P, V q) {
  V c1 = v3(M[0], M[1], M[2]), c2 = v3(M[3], M[4], M[5]);
  int tex = (int)M[13]; float sc = M[14], amp = M[15];
  if (tex == 0) return c1;
  if (tex == 1) { float k = fbm(q * sc, 4) * 0.5f + 0.5f; return c1 * (1 - amp + 2 * amp * k); }
  if (tex == 2) {
    float k = powf(clamp01(fbm(q * sc, 5) * 0.5f + 0.5f), 1.5f);
    float grain = vnoise(q * (sc * 6)) * 0.5f + 0.5f;
    float spot = vnoise(q * (sc * 2.5f) + v3(7, 7, 7)) > 0.6f ? 1.6f : 1.f;
    return lerp(c1, c2, k) * ((0.75f + 0.5f * grain) * spot);
  }
  if (tex == 3) {
    float r = sqrtf(q.y * q.y + q.z * q.z) * sc + 3.f * fbm(q * 8.f, 3);
    float g = powf(sinf(r * 6.283f) * 0.5f + 0.5f, 3.f);
    return lerp(c1, c2, g);
  }
  if (tex == 4) { float k = vnoise(v3(q.x * 900.f, q.y * 6.f, q.z * 6.f)) * 0.5f + 0.5f; return c1 * (0.85f + 0.3f * k); }
  if (tex == 5) { float k = fbm(q * sc, 3) * 0.5f + 0.5f; return c1 * (0.9f + 0.2f * k); }
  if (tex == 6) {
    bool band = fabsf(q.x) < amp;
    float st = sinf(atan2f(q.y, q.z) * 7.f) > 0.6f ? 1.f : 0.7f;
    return band ? c2 * st : c1;
  }
  if (tex == 7) {
    float ang = atan2f(q.y, q.x), rr = sqrtf(q.x * q.x + q.y * q.y);
    float k = vnoise(v3(ang * 60.f, rr * 40.f, ang * 3.f)) * 0.5f + 0.5f;
    return c1 * (0.5f + 0.9f * k);
  }
  // 8: robot body — white panels, red trim on edges, black star and red dot on the front
  float hx = P[16], hy = P[17], hz = P[18], e = 0.04f;
  int nearc = (fabsf(q.x) > hx - e) + (fabsf(q.y) > hy - e) + (fabsf(q.z) > hz - e);
  if (nearc >= 2) return c2;
  if (q.x > hx - 0.006f) {
    if (star_dist(q.y + 0.01f, q.z - 0.02f, 0.085f, 0.006f) < 0.f) return v3(0.018f, 0.018f, 0.02f);
    if (sqrtf((q.y + 0.12f) * (q.y + 0.12f) + (q.z - 0.06f) * (q.z - 0.06f)) < 0.012f) return c2;
  }
  return c1;
}

// ------------------------------------------------------------------ shading
struct Cam { float fx, fy, cx, cy; int W, H; V o; float R[9]; };
__device__ V cam_ray(const Cam& c, float px, float py, float& zf) {
  V dc = v3((px - c.cx) / c.fx, (py - c.cy) / c.fy, 1.f);
  V dw = v3(c.R[0] * dc.x + c.R[1] * dc.y + c.R[2] * dc.z, c.R[3] * dc.x + c.R[4] * dc.y + c.R[5] * dc.z,
            c.R[6] * dc.x + c.R[7] * dc.y + c.R[8] * dc.z);
  float L = len(dw); zf = 1.f / L; return dw * (1.f / L);
}
__device__ bool cam_project(const Cam& c, V p, float& px, float& py) {
  V d = p - c.o;
  float x = c.R[0] * d.x + c.R[3] * d.y + c.R[6] * d.z, y = c.R[1] * d.x + c.R[4] * d.y + c.R[7] * d.z;
  float z = c.R[2] * d.x + c.R[5] * d.y + c.R[8] * d.z;
  if (z < 1e-4f) return false;
  px = x / z * c.fx + c.cx; py = y / z * c.fy + c.cy; return true;
}

struct Env {       // mode 0: CG floor; mode 1: real background photo
  int mode; Img floor; float sx, sy; V room;
  Img bg_s, bg_l; const float* bgfloor; V fallback;
};
__device__ V env_lookup(const Env& E, const Cam& C, V p, V r, float rough) {
  if (r.z >= -1e-3f) return E.room * (0.8f + 0.4f * clamp01(r.z));
  float t = -p.z / r.z; V q = p + r * t;
  if (E.mode == 0) return floor_albedo(E.floor, E.sx, E.sy, q.x, q.y, 0.004f + rough * 0.04f + t * 0.01f) * 0.8f;
  float px, py;
  if (!cam_project(C, q, px, py) || px < 0 || py < 0 || px > C.W - 1 || py > C.H - 1) return E.fallback;
  return lerp(tex_fetch(E.bg_s, 0, px, py), tex_fetch(E.bg_l, 0, px, py), rough);
}
__device__ V ggx(V n, V v, V l, float rough, V f0) {
  V h = nrm(v + l);
  float nl = fmaxf(dot(n, l), 0.f), nv = fmaxf(dot(n, v), 1e-3f), nh = fmaxf(dot(n, h), 0.f), vh = fmaxf(dot(v, h), 0.f);
  float a = fmaxf(rough * rough, 0.02f), a2 = a * a;
  float dd = nh * nh * (a2 - 1.f) + 1.f, D = a2 / (PI * dd * dd);
  float k = (rough + 1.f) * (rough + 1.f) / 8.f;
  float G = nl / (nl * (1 - k) + k) * nv / (nv * (1 - k) + k);
  float fr = powf(1.f - vh, 5.f);
  V F = f0 + (v3(1, 1, 1) - f0) * fr;
  return F * (D * G / (4.f * nv) * nl);
}
__device__ V shade(const float* prims, int n, const float* mats, const Env& E, const Cam& C, V o, V d, float t,
                   int prim, V irr, V light, float jit, float& spec_out, bool allow_glass) {
  V p = o + d * t, nn = normal_at(prims, n, p), v = -d;
  const float* P = prims + prim * PS;
  const float* M = mats + (int)P[1] * MS;
  V q = to_local(P, p);
  V alb = albedo(M, P, q);
  V lv = light - p; float ld = len(lv); V l = lv * (1.f / ld);
  float upl = fmaxf(l.z, 0.05f);
  float sh = soft_shadow(prims, n, p + nn * 2e-3f, l, fminf(ld, 3.f), 12.f, jit);
  float ao = ao_at(prims, n, p, nn, 0.03f);
  V iamb = irr * 0.45f, idir = irr * (0.55f / upl);
  float rough = M[6], metal = M[7], trans = M[8];
  float nl = fmaxf(dot(nn, l), 0.f);
  V diff = mul(alb, iamb * ao + idir * (nl * sh));
  V f0 = v3(0.04f, 0.04f, 0.04f) * (1 - metal) + alb * metal;
  V sp = mul(ggx(nn, v, l, rough, f0), idir) * (sh * 1.4f);
  float cv = clamp01(dot(nn, v)), fr = powf(1.f - cv, 5.f);
  V fres = f0 + (v3(1, 1, 1) - f0) * fr;
  V r = nrm(reflect(d, nn));
  V env = env_lookup(E, C, p + nn * 1e-3f, r, rough) * ao;
  float gloss = (1 - rough) * (1 - rough);
  V col = diff * ((1 - metal) * (1 - trans)) + sp + mul(env, fres) * (gloss * (1 - metal) + metal) + env * (0.04f * M[16]);
  if (trans > 0.f && allow_glass) {
    V t1 = refract_dir(d, nn, 1.f / M[9]);
    V o2 = p - nn * 2e-3f; float tt = 1e-3f; int pr2 = -1;
    march(prims, n, o2, t1, 0.5f, 48, tt, pr2, -1.f);
    V q2 = o2 + t1 * tt, n2 = normal_at(prims, n, q2);
    V t2 = refract_dir(t1, -n2, M[9]);
    float th = fminf(tt, 0.2f) / 0.005f;
    V tint = v3(powf(M[10], th), powf(M[11], th), powf(M[12], th));
    V behind = env_lookup(E, C, q2 + t2 * 2e-3f, t2, 0.05f);
    float fm = (fres.x + fres.y + fres.z) / 3.f;
    col = col + mul(behind, tint) * ((1.f - fm) * trans);
  }
  V h = nrm(l + v);
  spec_out = powf(fmaxf(dot(nn, h), 0.f), 60.f) * sh;
  return col;
}

__device__ __forceinline__ float rnd(unsigned int& s) { s = s * 1664525u + 1013904223u; return (s >> 8) * (1.f / 16777216.f); }

// ------------------------------------------------------------------ kernel
extern "C" __global__ void render(
    const float* prims, int nprim, const float* mats,
    float fx, float fy, float cx, float cy, int W, int H, float ox, float oy, float oz, const float* R,
    int mode, const float* floor_tex, int fw, int fh, int flevels, float fsx, float fsy, float fgloss,
    const float* bg, const float* bg_s, const float* bg_l, const float* bg_depth, const float* bg_floor,
    float roomr, float roomg, float roomb, float lx, float ly, float lz, float irr_r, float irr_g, float irr_b,
    int spp, unsigned int seed,
    float* out_rgb, float* out_depth, int* out_id, float* out_spec, float* out_cov) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  int pix = y * W + x;
  Cam C; C.fx = fx; C.fy = fy; C.cx = cx; C.cy = cy; C.W = W; C.H = H; C.o = v3(ox, oy, oz);
  for (int i = 0; i < 9; i++) C.R[i] = R[i];
  Env E; E.mode = mode; E.floor.p = floor_tex; E.floor.w = fw; E.floor.h = fh; E.floor.levels = flevels;
  E.sx = fsx; E.sy = fsy; E.room = v3(roomr, roomg, roomb);
  E.bg_s.p = bg_s; E.bg_s.w = W; E.bg_s.h = H; E.bg_s.levels = 1;
  E.bg_l.p = bg_l; E.bg_l.w = W; E.bg_l.h = H; E.bg_l.levels = 1;
  E.bgfloor = bg_floor; E.fallback = E.room;
  V light = v3(lx, ly, lz), irr0 = v3(irr_r, irr_g, irr_b);
  unsigned int rs = seed ^ (pix * 9781u + 6271u);

  V bgc = v3(0, 0, 0); float bgz = 0.f; bool bgflo = true;
  if (mode == 1) {
    bgc = v3(bg[pix * 3], bg[pix * 3 + 1], bg[pix * 3 + 2]); bgz = bg_depth[pix]; bgflo = bg_floor[pix] > 0.5f || bgz == 0.f;
  }
  V acc = v3(0, 0, 0); float cover = 0.f, zbest = 1e3f, specbest = 0.f; int idbest = -1;
  V floor_acc = v3(0, 0, 0);
  for (int s = 0; s < spp; s++) {
    float jx = spp > 1 ? ((s & 1) ? 0.75f : 0.25f) : 0.5f, jy = spp > 1 ? ((s & 2) ? 0.75f : 0.25f) : 0.5f;
    float zf; V d = cam_ray(C, x + jx - 0.5f, y + jy - 0.5f, zf);
    V o = C.o;
    float tfl = d.z < -1e-4f ? -o.z / d.z : 1e3f;
    float te = sentry(prims, nprim, o, d);
    float t = te; int prim = -1; bool hit = false;
    if (te < tfl && te < 1e29f) hit = march(prims, nprim, o, d, fminf(tfl + 0.01f, 12.f), 160, t, prim, 1.f);
    float zc = t * zf;
    bool visible = hit && (mode == 0 ? t < tfl : (bgz == 0.f || zc < bgz + 0.004f || bgflo));
    float jit = rnd(rs);
    if (visible) {
      V irr = irr0;
      if (mode == 1) {                       // irradiance from the real floor next to the hit
        V hp = o + d * t; float px, py;
        if (cam_project(C, v3(hp.x, hp.y, 0.f), px, py)) irr = tex_fetch(E.bg_l, 0, px, py) * 2.f;
      }
      float sp = 0.f;
      V c = shade(prims, nprim, mats, E, C, o, d, t, prim, irr, light, jit, sp, true);
      acc = acc + c; cover += 1.f;
      if (zc < zbest) { zbest = zc; idbest = (int)prims[prim * PS + 2]; }
      specbest = fmaxf(specbest, sp);
    } else if (mode == 0 && tfl < 1e3f) {       // CG floor
      V P = o + d * tfl;
      float foot = tfl / C.fx / fmaxf(fabsf(d.z), 0.05f);
      V alb = floor_albedo(E.floor, fsx, fsy, P.x, P.y, foot);
      V l = nrm(light - P);
      float sh = 1.f, ao = 1.f;
      // only near objects: shadows, contact occlusion, reflections
      bool nearobj = false;
      for (int i = 0; i < nprim && !nearobj; i++) {
        const float* Q = prims + i * PS;
        float dx = P.x - Q[4], dy = P.y - Q[5];
        nearobj = sqrtf(dx * dx + dy * dy) < Q[28] * 1.6f + 0.25f;
      }
      if (nearobj) {
        sh = soft_shadow(prims, nprim, P + v3(0, 0, 1e-3f), l, 3.f, 10.f, jit);
        ao = ao_at(prims, nprim, P, v3(0, 0, 1), 0.05f);
      }
      V c = mul(alb, irr0) * (0.45f * ao + 0.55f * sh);
      c = c + mul(ggx(v3(0, 0, 1), -d, l, 0.25f, v3(0.04f, 0.04f, 0.04f)), irr0) * (3.f * sh);
      if (nearobj) {
        V rd = v3(d.x, d.y, -d.z);
        float re = sentry(prims, nprim, P, rd);
        if (re < 2.f) {
          float tr = re + jit * 0.004f; int pr = -1;
          if (march(prims, nprim, P, rd, 2.f, 110, tr, pr, 1.f)) {
            const float* Q = prims + pr * PS; const float* M = mats + (int)Q[1] * MS;
            V rc = mul(v3(M[0], M[1], M[2]), irr0) * 0.6f;
            float w = fgloss * expf(-tr * 3.f);
            c = lerp(c, rc, w);
          }
        }
      }
      float fog = expf(-fmaxf(tfl - 3.f, 0.f) * 0.15f);
      floor_acc = floor_acc + lerp(E.room, c, fog);
    } else if (mode == 0) {
      floor_acc = floor_acc + E.room * (0.8f + 0.4f * clamp01(d.z));
    }
  }
  float a = cover / spp;
  V obj = cover > 0.f ? acc * (1.f / cover) : v3(0, 0, 0);
  V base;
  if (mode == 0) {
    base = spp - cover > 0.f ? floor_acc * (1.f / (spp - cover)) : v3(0, 0, 0);
  } else {
    base = bgc;
    if (bgflo) {                               // real floor near objects: shadow, occlusion, glossy reflection
      float zf; V d = cam_ray(C, x, y, zf);
      if (d.z < -1e-4f) {
        V P = C.o + d * (-C.o.z / d.z);
        bool nearobj = false;
        for (int i = 0; i < nprim && !nearobj; i++) {
          const float* Q = prims + i * PS;
          float dx = P.x - Q[4], dy = P.y - Q[5];
          nearobj = sqrtf(dx * dx + dy * dy) < Q[28] * 3.f + 0.06f;
        }
        if (nearobj) {
          V l = nrm(light - P);
          float sh = soft_shadow(prims, nprim, P + v3(0, 0, 1e-3f), l, 2.5f, 10.f, rnd(rs));
          float ao = ao_at(prims, nprim, P, v3(0, 0, 1), 0.04f);
          base = base * (0.45f * ao + 0.55f * sh);
          V rd = v3(d.x, d.y, -d.z);
          float re = sentry(prims, nprim, P, rd);
          if (re < 1.5f) {
            float tr = re + rnd(rs) * 0.004f; int pr = -1;
            if (march(prims, nprim, P, rd, 1.5f, 110, tr, pr, 1.f)) {
              const float* Q = prims + pr * PS; const float* M = mats + (int)Q[1] * MS;
              float px, py; V irr = irr0;
              if (cam_project(C, P, px, py)) irr = tex_fetch(E.bg_l, 0, px, py) * 2.f;
              V rc = mul(v3(M[0], M[1], M[2]), irr) * 0.6f;
              base = lerp(base, rc, 0.10f * expf(-tr * 6.f));
            }
          }
        }
      }
    }
  }
  V col = lerp(base, obj, a);
  out_rgb[pix * 3] = col.x; out_rgb[pix * 3 + 1] = col.y; out_rgb[pix * 3 + 2] = col.z;
  out_cov[pix] = a;
  if (a >= 0.5f) { out_depth[pix] = zbest; out_id[pix] = idbest; out_spec[pix] = specbest; }
  else { out_depth[pix] = mode == 1 ? bgz : 0.f; out_id[pix] = -1; out_spec[pix] = 0.f; }
}

// distance field on arbitrary points (volume integration, placement checks)
extern "C" __global__ void sdf_points(const float* prims, int nprim, const float* pts, int npts, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npts) return;
  out[i] = scene(prims, nprim, v3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]), false).d;
}
