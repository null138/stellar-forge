import random, sys, threading
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import numpy as np
from numba import njit, prange
import tkinter as tk
from tkinter import ttk, filedialog
import ctypes

PREVIEW_W = 1280
PREVIEW_H = 720

def blend_color_dodge(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
	if top.ndim == 2:
		top = top[..., None]
	if top.shape[-1] == 1 and bottom.shape[-1] == 3:
		top = np.repeat(top, 3, axis=2)
	denom = np.clip(1.0 - top, 1e-6, 1.0)
	return np.clip(bottom / denom, 0.0, 1.0)

def apply_levels(arr: np.ndarray, in_black: float, gamma: float, in_white: float) -> np.ndarray:
	lo = in_black / 255.0
	hi = in_white / 255.0
	arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
	return np.power(arr, 1.0 / gamma).astype(np.float32)

def _equirect_to_sphere(width: int, height: int) -> tuple:
	lon = np.linspace(0.0, 2.0 * np.pi, width,	endpoint=False, dtype=np.float64)
	lat = np.linspace(0.0,		 np.pi, height, endpoint=True,	dtype=np.float64)
	lon2d, lat2d = np.meshgrid(lon, lat)
	sx = np.sin(lat2d) * np.cos(lon2d)
	sy = np.sin(lat2d) * np.sin(lon2d)
	sz = np.cos(lat2d)
	return sx.astype(np.float32), sy.astype(np.float32), sz.astype(np.float32)

# not my code and i dont want to comment anything about it.
# sadly its still a cpu work. handling it on gpu would require a lot of gpu brand compability checks.
@njit(cache=True)
def _fade(t):
	return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

@njit(cache=True)
def _lerp(a, b, t):
	return a + t * (b - a)

@njit(cache=True)
def _hash2d(ix, iy, seed):
	h = ix * 374761393 + iy * 668265263 + seed * 1013904223
	h ^= h >> 16
	h *= 0x85ebca6b
	h ^= h >> 13
	h *= 0xc2b2ae35
	h ^= h >> 16
	return h

@njit(cache=True)
def _grad(hash_val, x, y):
	h = hash_val & 3
	u = x if h < 2 else y
	v = y if h < 2 else x
	return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

@njit(cache=True)
def _perlin2d(x, y, seed):
	xi = int(np.floor(x)); yi = int(np.floor(y))
	xf = x - xi;		   yf = y - yi
	u = _fade(xf);			v = _fade(yf)
	aa = _hash2d(xi,	 yi,	 seed)
	ba = _hash2d(xi + 1, yi,	 seed)
	ab = _hash2d(xi,	 yi + 1, seed)
	bb = _hash2d(xi + 1, yi + 1, seed)
	x1 = _lerp(_grad(aa, xf,	 yf),	  _grad(ba, xf - 1, yf),	 u)
	x2 = _lerp(_grad(ab, xf,	 yf - 1), _grad(bb, xf - 1, yf - 1), u)
	return _lerp(x1, x2, v)

@njit(cache=True)
def _perlin2d_octaves(x, y, octaves, seed):
	total = 0.0; amplitude = 1.0; frequency = 1.0; max_amp = 0.0
	for _ in range(octaves):
		total	+= amplitude * _perlin2d(x * frequency, y * frequency, seed)
		max_amp += amplitude
		amplitude *= 0.5
		frequency *= 2.0
	return total / max_amp

@njit(parallel=True, cache=True)
def _perlin_array(xv, yv, octaves, seed):
	h, w = xv.shape
	out	 = np.empty((h, w), dtype=np.float32)
	for i in prange(h):
		for j in range(w):
			out[i, j] = _perlin2d_octaves(xv[i, j], yv[i, j], octaves, seed)
	return out

def perlin_field(width, height, scale, octaves, ox, oy, seed=0):
	REF_W, REF_H = 800.0, 450.0
	x  = np.linspace(ox, ox + REF_W / scale, width,	 dtype=np.float64)
	y  = np.linspace(oy, oy + REF_H / scale, height, dtype=np.float64)
	xv, yv = np.meshgrid(x, y)
	arr = _perlin_array(xv, yv, octaves, seed)
	arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
	return arr

@njit(cache=True)
def _hash3d(ix, iy, iz, seed):
	h = ix * 374761393 + iy * 668265263 + iz * 1274126177 + seed * 1013904223
	h ^= h >> 16
	h *= 0x85ebca6b
	h ^= h >> 13
	h *= 0xc2b2ae35
	h ^= h >> 16
	return h

@njit(cache=True)
def _grad3(hash_val, x, y, z):
	h = hash_val & 15
	u = x if h < 8 else y
	v = y if h < 4 else (x if h == 12 or h == 14 else z)
	return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

@njit(cache=True)
def _perlin3d(x, y, z, seed):
	xi = int(np.floor(x)); yi = int(np.floor(y)); zi = int(np.floor(z))
	xf = x - xi; yf = y - yi; zf = z - zi
	u = _fade(xf); v = _fade(yf); w = _fade(zf)
	aaa = _hash3d(xi,	yi,	  zi,	seed)
	baa = _hash3d(xi+1, yi,	  zi,	seed)
	aba = _hash3d(xi,	yi+1, zi,	seed)
	bba = _hash3d(xi+1, yi+1, zi,	seed)
	aab = _hash3d(xi,	yi,	  zi+1, seed)
	bab = _hash3d(xi+1, yi,	  zi+1, seed)
	abb = _hash3d(xi,	yi+1, zi+1, seed)
	bbb = _hash3d(xi+1, yi+1, zi+1, seed)
	x1 = _lerp(_grad3(aaa, xf,	 yf,   zf),	  _grad3(baa, xf-1, yf,	  zf),	 u)
	x2 = _lerp(_grad3(aba, xf,	 yf-1, zf),	  _grad3(bba, xf-1, yf-1, zf),	 u)
	y1 = _lerp(x1, x2, v)
	x1 = _lerp(_grad3(aab, xf,	 yf,   zf-1), _grad3(bab, xf-1, yf,	  zf-1), u)
	x2 = _lerp(_grad3(abb, xf,	 yf-1, zf-1), _grad3(bbb, xf-1, yf-1, zf-1), u)
	y2 = _lerp(x1, x2, v)
	return _lerp(y1, y2, w)

@njit(cache=True)
def _perlin3d_octaves(x, y, z, octaves, seed):
	total = 0.0; amplitude = 1.0; frequency = 1.0; max_amp = 0.0
	for _ in range(octaves):
		total	+= amplitude * _perlin3d(x * frequency, y * frequency, z * frequency, seed)
		max_amp += amplitude
		amplitude *= 0.5
		frequency *= 2.0
	return total / max_amp

@njit(parallel=True, cache=True)
def _perlin3d_array(sx, sy, sz, scale, octaves, seed):
	h, w = sx.shape
	out	 = np.empty((h, w), dtype=np.float32)
	for i in prange(h):
		for j in range(w):
			out[i, j] = _perlin3d_octaves(
				sx[i, j] * scale, sy[i, j] * scale, sz[i, j] * scale,
				octaves, seed)
	return out

# early version template. decided to leave it.
palettes = {
	"warm":	 [(255, 80,	 30),  (255, 160, 20),	(200, 40,  80),	 (255, 200, 80)],
	"cool":	 [(30,	80,	 255), (20,	 200, 255), (80,  30,  200), (40,  180, 220)],
	"mixed": [(180, 20,	 255), (20,	 220, 180), (255, 60,  120), (60,  120, 255)],
}

def make_L0(width: int, height: int, rng, density=0.5, scale=0.5) -> np.ndarray:
	noise = np.abs(rng.normal(0.0, density, (height, width))).astype(np.float32)
	noise = np.clip(noise, 0.0, 1.0)
	img = Image.fromarray((noise * 255).astype(np.uint8), "L")
	img = img.filter(ImageFilter.GaussianBlur(radius=scale))
	blurred = np.array(img).astype(np.float32) / 255.0
	blurred = apply_levels(blurred, in_black=170, gamma=0.40, in_white=255)
	return np.stack([blurred, blurred, blurred], axis=-1)

def make_L1(width: int, height: int, rng, density=0.6, scale=1.5) -> np.ndarray:
	noise = np.abs(rng.normal(0.0, density, (height, width))).astype(np.float32)
	noise = np.clip(noise, 0.0, 1.0)
	img = Image.fromarray((noise * 255).astype(np.uint8), "L")
	img = img.filter(ImageFilter.GaussianBlur(radius=scale))
	blurred = np.array(img).astype(np.float32) / 255.0
	blurred = apply_levels(blurred, in_black=170, gamma=1.0, in_white=255)
	return np.stack([blurred, blurred, blurred], axis=-1)

def make_L2(width: int, height: int, rng, custom_colors=None) -> np.ndarray:
	if custom_colors:
		active_palette = custom_colors
	else:
		key = rng.choice(["warm", "cool", "mixed"])
		active_palette = palettes[key]
	REF_W, REF_H = 800, 450
	num_blobs = int(rng.integers(3, 8))
	img	 = Image.fromarray(np.zeros((REF_H, REF_W, 3), dtype=np.uint8), "RGB")
	draw = ImageDraw.Draw(img)
	for _ in range(num_blobs):
		base  = active_palette[rng.integers(0, len(active_palette))]
		color = tuple(max(0, min(255, c + int(rng.integers(-30, 31)))) for c in base)
		cx = int(rng.uniform(0, REF_W))
		cy = int(rng.uniform(0, REF_H))
		rx = int(rng.uniform(REF_W / 6.0, REF_W / 2.0))
		ry = int(rng.uniform(REF_H / 6.0, REF_H / 2.0))
		draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
	img = img.filter(ImageFilter.GaussianBlur(radius=80))
	img = img.resize((width, height), Image.BILINEAR)
	arr	 = np.array(img).astype(np.float32) / 255.0
	arr *= 0.30
	return arr

def make_L3(width, height, rng):
	scale	= rng.uniform(250, 450)
	octaves = 6
	ox1, oy1 = float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))
	ox2, oy2 = float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))
	p1	 = perlin_field(width, height, scale, octaves, ox1, oy1, seed=42)
	p2	 = perlin_field(width, height, scale, octaves, ox2, oy2, seed=137)
	base = np.abs(np.abs(0.5 - p1) - p2)
	base = (base - base.min()) / (base.max() - base.min() + 1e-6)
	return base

def _sample_canvas_sphere(canvas: np.ndarray, sx, sy, sz) -> np.ndarray:
	cvs_h, cvs_w = canvas.shape[:2]
	is_color = canvas.ndim == 3

	lon = (np.arctan2(sy, sx) % (2.0 * np.pi)).astype(np.float32)
	lat = np.arccos(np.clip(sz, -1.0, 1.0)).astype(np.float32)

	cx = lon / (2.0 * np.pi) * cvs_w
	cy = lat / np.pi		  * cvs_h

	x0 = np.floor(cx).astype(np.int32) % cvs_w
	x1 = (x0 + 1) % cvs_w
	y0 = np.clip(np.floor(cy).astype(np.int32), 0, cvs_h - 1)
	y1 = np.clip(y0 + 1,						0, cvs_h - 1)
	xf = (cx - np.floor(cx)).astype(np.float32)
	yf = (cy - np.floor(cy)).astype(np.float32)

	if is_color:
		xf = xf[..., None]; yf = yf[..., None]

	out = (canvas[y0, x0] * (1 - xf) * (1 - yf) +
		   canvas[y0, x1] *		 xf	 * (1 - yf) +
		   canvas[y1, x0] * (1 - xf) *		yf	+
		   canvas[y1, x1] *		 xf	 *		yf).astype(np.float32)
	return out


def _make_star_layer_sphere(width: int, height: int, rng,density: float, blur_radius: float, gamma: float, sphere_coords) -> np.ndarray:
	sx, sy, sz = sphere_coords
	cvs_w = 800
	cvs_h = 450
	tile_w = cvs_w * 3
	noise = np.abs(rng.normal(0.0, density, (cvs_h, cvs_w))).astype(np.float32)
	noise = np.clip(noise, 0.0, 1.0)
	tiled = np.tile(noise, (1, 3))
	img = Image.fromarray((tiled * 255).astype(np.uint8), "L")
	img = img.filter(ImageFilter.GaussianBlur(radius=max(blur_radius, 0.4)))
	blurred_tiled = np.array(img).astype(np.float32) / 255.0
	canvas = blurred_tiled[:, cvs_w: cvs_w * 2]
	sampled = _sample_canvas_sphere(canvas, sx, sy, sz)
	sampled = apply_levels(sampled, in_black=170, gamma=gamma, in_white=255)
	return np.stack([sampled, sampled, sampled], axis=-1)


def make_L0_sphere(width: int, height: int, rng, density=0.5, scale=0.5, sphere_coords=None) -> np.ndarray:
	if sphere_coords is None:
		sphere_coords = _equirect_to_sphere(width, height)
	return _make_star_layer_sphere(width, height, rng,density, max(scale * 0.5, 0.4), 0.40, sphere_coords)

def make_L1_sphere(width: int, height: int, rng, density=0.6, scale=1.5, sphere_coords=None) -> np.ndarray:
	if sphere_coords is None:
		sphere_coords = _equirect_to_sphere(width, height)
	return _make_star_layer_sphere(width, height, rng, density, max(scale * 1.0, 0.5), 1.0, sphere_coords)

def make_L2_sphere(width: int, height: int, rng, custom_colors=None, sphere_coords=None) -> np.ndarray:
	if sphere_coords is None:
		sphere_coords = _equirect_to_sphere(width, height)
	sx, sy, sz = sphere_coords

	if custom_colors:
		active_palette = custom_colors
	else:
		key = rng.choice(["warm", "cool", "mixed"])
		active_palette = palettes[key]

	REF_W, REF_H = 800, 450
	tile_w = REF_W * 3
	img	 = Image.fromarray(np.zeros((REF_H, tile_w, 3), dtype=np.uint8), "RGB")
	draw = ImageDraw.Draw(img)
	num_blobs = int(rng.integers(3, 8))
	for _ in range(num_blobs):
		base  = active_palette[rng.integers(0, len(active_palette))]
		color = tuple(max(0, min(255, c + int(rng.integers(-30, 31)))) for c in base)
		cx = int(rng.uniform(REF_W, REF_W * 2))
		cy = int(rng.uniform(0, REF_H))
		rx = int(rng.uniform(REF_W / 6.0, REF_W / 2.0))
		ry = int(rng.uniform(REF_H / 6.0, REF_H / 2.0))
		for dx in [-REF_W, 0, REF_W]:
			draw.ellipse([cx + dx - rx, cy - ry, cx + dx + rx, cy + ry], fill=color)

	img = img.filter(ImageFilter.GaussianBlur(radius=80))
	tiled = np.array(img).astype(np.float32) / 255.0
	canvas = tiled[:, REF_W: REF_W * 2]
	result = _sample_canvas_sphere(canvas, sx, sy, sz)
	result = np.clip(result, 0.0, 1.0)
	result *= 0.30
	return result


def make_L3_sphere(width: int, height: int, rng, sphere_coords=None) -> np.ndarray:
	if sphere_coords is None:
		sphere_coords = _equirect_to_sphere(width, height)
	sx, sy, sz = sphere_coords
	scale	= float(rng.uniform(1.5, 2.5))
	octaves = 6
	seed1	= int(rng.integers(0, 2**31))
	seed2	= int(rng.integers(0, 2**31))
	p1 = _perlin3d_array(sx, sy, sz, scale, octaves, seed1)
	p2 = _perlin3d_array(sx, sy, sz, scale, octaves, seed2)
	p1 = (p1 - p1.min()) / (p1.max() - p1.min() + 1e-6)
	p2 = (p2 - p2.min()) / (p2.max() - p2.min() + 1e-6)
	base = np.abs(np.abs(0.5 - p1) - p2)
	base = (base - base.min()) / (base.max() - base.min() + 1e-6)
	return base.astype(np.float32)


class SpaceGUI:
	def _make_slider(self, parent, label, var, frm, to, on_change):
		BG	   = "#1e1f22"
		ACCENT = "#00e5ff"
		row = ttk.Frame(parent)
		row.pack(fill="x", pady=1)
		top = tk.Frame(row, bg=BG)
		top.pack(fill="x")
		ttk.Label(top, text=label).pack(side="left")
		value_label = tk.Label(top, text=f"{var.get():.2f}",
							   fg=ACCENT, bg=BG,
							   font=("Segoe UI", 8),
							   cursor="xterm")
		value_label.pack(side="right")
		edit_var	= tk.StringVar()
		value_entry = tk.Entry(top, textvariable=edit_var, width=5,
							   fg=ACCENT, bg="#2a2d31",
							   insertbackground=ACCENT,
							   relief="flat", font=("Segoe UI", 8),
							   highlightthickness=1,
							   highlightcolor=ACCENT,
							   highlightbackground="#3a3d42",
							   justify="right")

		def _show_entry(event=None):
			edit_var.set(f"{var.get():.2f}")
			value_label.pack_forget()
			value_entry.pack(side="right")
			value_entry.focus_set()
			value_entry.select_range(0, "end")

		def _hide_entry():
			value_entry.pack_forget()
			value_label.pack(side="right")

		def _commit(event=None):
			try:
				v = float(edit_var.get())
				v = max(frm, min(to, v))
				var.set(v)
				value_label.config(text=f"{v:.2f}")
				on_change(str(v))
			except ValueError:
				pass
			_hide_entry()

		def _cancel(event=None):
			_hide_entry()

		value_label.bind("<Button-1>", _show_entry)
		value_entry.bind("<Return>",   _commit)
		value_entry.bind("<KP_Enter>", _commit)
		value_entry.bind("<Escape>",   _cancel)
		value_entry.bind("<FocusOut>", _commit)

		def _slider_moved(val):
			value_label.config(text=f"{float(val):.2f}")
			on_change(val)

		ttk.Scale(row, from_=frm, to=to, variable=var,
				  style="Red.Horizontal.TScale",
				  command=_slider_moved).pack(fill="x", expand=True)

	def slider(self, parent, label, var, frm, to):
		self._make_slider(parent, label, var, frm, to,
						  lambda val: self._schedule_render())

	def slider_l0(self, parent, label, var, frm, to):
		self._make_slider(parent, label, var, frm, to,
						  lambda val: self._schedule_rebuild_l0())

	def slider_l1(self, parent, label, var, frm, to):
		self._make_slider(parent, label, var, frm, to,
						  lambda val: self._schedule_rebuild_l1())

	def _drag_start(self, event):
		self._drag_x = event.x_root - self.root.winfo_x()
		self._drag_y = event.y_root - self.root.winfo_y()

	def _drag_move(self, event):
		self.root.geometry(f"+{event.x_root - self._drag_x}+{event.y_root - self._drag_y}")

	def _minimize(self):
		self.root.iconify()

	def get_render_size(self):
		try:
			w = int(self.res_width_var.get())
			h = int(self.res_height_var.get())
			if w < 1 or h < 1:
				raise ValueError
			return w, h
		except ValueError:
			return 1920, 1080

	def make_borderless(self):
		self.root.update_idletasks()
		hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
		GWL_STYLE = -16
		style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
		style = style & ~0x00C00000 & ~0x00040000
		ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)
		ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0020 | 0x0001 | 0x0002 | 0x0040)

	def apply_seed(self):
		try:
			self.seed = int(self.seed_var.get())
		except ValueError:
			self.seed = random.randint(0, 10**12)
			self.seed_var.set(str(self.seed))
		self.generate_cache()
		self.render_from_cache()

	def _cubemap_mode(self) -> bool:
		return bool(self.cubemap_var.get())

	def __init__(self, root):
		self.root = root
		self.root.title("Stellar Forge v1.2 by Madness (null138)")
		self.border_width = 5
		self._render_pending = False
		self._render_after = None
		self._rebuild_l0_after = None
		self._rebuild_l1_after = None
		self._busy = False
		self.bg_canvas = tk.Canvas(root, highlightthickness=0, bd=0, width=1600, height=850)
		self.bg_canvas.pack(fill="both", expand=True)
		self.main_ui = tk.Frame(self.bg_canvas, bg="#1e1f22")
		self.main_ui.place(relx=0, rely=0, relwidth=1, relheight=1,
						   x=self.border_width, y=self.border_width,
						   width=-self.border_width * 2, height=-self.border_width * 2)
		self.root.update_idletasks()
		self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

		def draw_gradient(event=None):
			w = self.bg_canvas.winfo_width()
			h = self.bg_canvas.winfo_height()
			self.bg_canvas.delete("grad")
			steps = 64
			for i in range(steps):
				r = int(255 - (255 * i / steps))
				g = int(0	+ (229 * i / steps))
				b = 255
				color = f'#{r:02x}{g:02x}{b:02x}'
				self.bg_canvas.create_rectangle(
					0, 0, w * (1 - i / steps), h * (1 - i / steps),
					fill=color, outline=color, tags="grad")
			self.bg_canvas.tag_lower("grad")
		self.bg_canvas.bind("<Configure>", draw_gradient)
		self.seed	  = random.randint(0, 10**12)
		self.seed_var = tk.StringVar(value=str(self.seed))
		self.l0_seed  = self.seed
		self.l1_seed  = self.seed + 1
		self.cached_l0 = None
		self.cached_l1 = None
		self.cached_l2 = None
		self.cached_l3 = None
		self._cached_sphere_coords = None
		self._cached_sphere_size   = (0, 0)
		self.fine_star		 = tk.DoubleVar(value=1.0)
		self.coarse_star	 = tk.DoubleVar(value=1.0)
		self.star_brightness = tk.DoubleVar(value=1.0)
		self.cloud_strength	 = tk.DoubleVar(value=1.0)
		self.cloud_contrast	 = tk.DoubleVar(value=1.0)
		self.wash_strength	 = tk.DoubleVar(value=1.0)
		self.vignette		 = tk.DoubleVar(value=0.0)
		self.brightness		 = tk.DoubleVar(value=1.0)
		self.l0_density		 = tk.DoubleVar(value=0.5)
		self.l0_scale		 = tk.DoubleVar(value=0.5)
		self.l1_density		 = tk.DoubleVar(value=0.6)
		self.l1_scale		 = tk.DoubleVar(value=1.5)
		self.res_width_var	 = tk.StringVar(value="1920")
		self.res_height_var	 = tk.StringVar(value="1080")
		self.cubemap_var	 = tk.IntVar(value=0)
		titlebar = tk.Frame(self.main_ui, bg="#2a2d31", height=28)
		titlebar.pack(side="top", fill="x")
		titlebar.pack_propagate(False)
		tk.Label(titlebar, text="Stellar Forge by Madness (null138)",
				 bg="#2a2d31", fg="#00e5ff", font=("Segoe UI", 9)).pack(side="left", padx=6)
		tk.Button(titlebar, text="✕", bg="#2a2d31", fg="#e6e6e6", bd=0,
				  activebackground="#e81123", activeforeground="white", width=3,
				  command=root.destroy).pack(side="right")
		tk.Button(titlebar, text="─", bg="#2a2d31", fg="#e6e6e6", bd=0,
				  activebackground="#3a3d42", activeforeground="white", width=3,
				  command=self._minimize).pack(side="right")
		titlebar.bind("<ButtonPress-1>", self._drag_start)
		titlebar.bind("<B1-Motion>",	 self._drag_move)
		style = ttk.Style()
		style.theme_use("clam")
		BG, PANEL, ACCENT, ACCENT_DARK = "#1e1f22", "#2a2d31", "#00e5ff", "#00a3b3"
		TEXT, MUTED = "#e6e6e6", "#a0a0a0"
		style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL)
		style.configure("TFrame", background=BG)
		style.configure("TLabel", background=BG, foreground=TEXT)
		style.configure("TLabelframe", background=BG, foreground=TEXT, borderwidth=0, relief="flat")
		style.configure("TLabelframe.Label", background=BG, foreground=ACCENT)
		style.configure("TButton", padding=4, background=PANEL, foreground=TEXT, borderwidth=0)
		style.map("TButton",
				  background=[("active", ACCENT_DARK)],
				  foreground=[("active", "#ffffff")])
		style.configure("Red.Horizontal.TScale",
						 background=BG, troughcolor=PANEL, sliderthickness=12,
						 lightcolor="#ff4d4d", darkcolor="#b30000", bordercolor="#ff1a1a")
		style.configure("TEntry", fieldbackground=PANEL, foreground=TEXT, insertcolor=ACCENT)
		style.configure("Cubemap.TCheckbutton",
						background=BG, foreground=ACCENT,
						focuscolor=BG, indicatorcolor=PANEL)
		controls_border = tk.Frame(self.main_ui, bg="#3a3d42", bd=0, relief="flat", padx=2, pady=2)
		controls_border.pack(side="left", fill="y", padx=5, pady=5)
		controls = ttk.Frame(controls_border, padding=5)
		controls.pack(fill="y", expand=True)

		stars_frame = ttk.LabelFrame(controls, text="Stars", padding=5)
		stars_frame.pack(fill="x", pady=4)
		stars_grid = ttk.Frame(stars_frame)
		stars_grid.pack(fill="x")

		fine_col = ttk.Frame(stars_grid)
		fine_col.grid(row=0, column=0, padx=5, sticky="nsew")
		self.slider(fine_col, "Fine", self.fine_star, 0.0, 3.0)
		ttk.Button(fine_col, text="🔄 Regenerate Fine", command=self.regen_l0).pack(fill="x", pady=1)
		self.slider_l0(fine_col, "Fine Density", self.l0_density, 0.0, 1.0)
		self.slider_l0(fine_col, "Fine Suppress", self.l0_scale, 0.0, 3.0)

		coarse_col = ttk.Frame(stars_grid)
		coarse_col.grid(row=0, column=1, padx=5, sticky="nsew")
		self.slider(coarse_col, "Coarse", self.coarse_star, 0.0, 3.5)
		ttk.Button(coarse_col, text="🔄 Regenerate Coarse", command=self.regen_l1).pack(fill="x", pady=1)
		self.slider_l1(coarse_col, "Coarse Density", self.l1_density, 0.0, 1.0)
		self.slider_l1(coarse_col, "Coarse Suppress", self.l1_scale, 0.0, 3.0)

		self.slider(stars_frame, "Brightness", self.star_brightness, 0.0, 2.0)

		clouds_frame = ttk.LabelFrame(controls, text="Clouds", padding=5)
		clouds_frame.pack(fill="x", pady=4)
		self.slider(clouds_frame, "Strength", self.cloud_strength, 0.0, 2.0)
		ttk.Button(clouds_frame, text="🔄 Regenerate Clouds", command=self.regen_l3).pack(fill="x", pady=1)
		self.slider(clouds_frame, "Contrast", self.cloud_contrast, 0.0, 3.0)

		color_frame = ttk.LabelFrame(controls, text="Color", padding=5)
		color_frame.pack(fill="x", pady=4)
		self.slider(color_frame, "Wash", self.wash_strength, 0.0, 2.0)
		ttk.Button(color_frame, text="🔄 Regenerate Color", command=self.regen_l2).pack(fill="x", pady=1)
		ttk.Label(color_frame, text="Custom colors (hex, comma):").pack(anchor="w")
		self.custom_colors_entry = ttk.Entry(color_frame)
		self.custom_colors_entry.pack(fill="x", pady=1)
		self.custom_colors_entry.insert(0, "")

		global_frame = ttk.LabelFrame(controls, text="Global", padding=5)
		global_frame.pack(fill="x", pady=4)
		self.slider(global_frame, "Vignette", self.vignette, 0.0, 2.0)
		self.slider(global_frame, "Brightness", self.brightness, 0.0, 3.0)

		res_frame = ttk.LabelFrame(controls, text="Resolution", padding=5)
		res_frame.pack(fill="x", pady=4)
		res_row = ttk.Frame(res_frame)
		res_row.pack(fill="x")
		ttk.Label(res_row, text="Width").pack(side="left")
		ttk.Entry(res_row, textvariable=self.res_width_var, width=5).pack(side="left", padx=(2, 5))
		ttk.Label(res_row, text="Height").pack(side="left")
		ttk.Entry(res_row, textvariable=self.res_height_var, width=5).pack(side="left", padx=(2, 0))

		ttk.Checkbutton(
			res_frame,
			text="Cubemap",
			variable=self.cubemap_var,
			style="Cubemap.TCheckbutton",
			command=self._on_cubemap_toggle,
		).pack(anchor="w", pady=(3, 0))

		btn_frame = ttk.Frame(controls)
		btn_frame.pack(fill="x", pady=5)
		seed_row = ttk.Frame(btn_frame)
		seed_row.pack(fill="x", pady=1)
		ttk.Entry(seed_row, textvariable=self.seed_var).pack(side="left", fill="x", expand=True, padx=(0, 4))
		ttk.Button(seed_row, text="Apply", command=self.apply_seed).pack(side="right")
		ttk.Button(btn_frame, text="Generate New seed", command=self.new_seed).pack(fill="x", pady=1)
		ttk.Button(btn_frame, text="Save", command=self.save_image).pack(fill="x", pady=1)

		canvas_border = tk.Frame(self.main_ui, bg="#3a3d42", bd=2, relief="flat")
		canvas_border.pack(side="right", expand=True, padx=5, pady=5)
		self.canvas = tk.Label(canvas_border, bg="#1e1f22", width=PREVIEW_W, height=PREVIEW_H)
		self.canvas.pack(padx=2, pady=2)

		self._status_var = tk.StringVar(value="")
		tk.Label(self.main_ui, textvariable=self._status_var, bg="#1e1f22", fg="#00e5ff",
				 font=("Segoe UI", 8)).pack(side="bottom", anchor="e", padx=10)

		self.image = None
		self.photo = None
		self._run_in_thread(self._generate_and_render_all)
		self.root.after(500, self.make_borderless)

	def _on_cubemap_toggle(self):
		self._run_in_thread(self._generate_and_render_all)

	def _get_sphere_coords(self, w, h):
		if self._cached_sphere_size != (w, h) or self._cached_sphere_coords is None:
			self._cached_sphere_coords = _equirect_to_sphere(w, h)
			self._cached_sphere_size   = (w, h)
		return self._cached_sphere_coords

	def _run_in_thread(self, fn, *args):
		if self._busy:
			self._render_pending = True
			return
		self._busy = True
		self.root.after(0, lambda: self._status_var.set("⏳ Rendering…"))
		def wrapper():
			try:
				fn(*args)
			finally:
				self._busy = False
				self.root.after(0, lambda: self._status_var.set(""))
		t = threading.Thread(target=wrapper, daemon=True)
		t.start()

	def _schedule_render(self):
		if self._render_after is not None:
			self.root.after_cancel(self._render_after)
		self._render_after = self.root.after(80, self._trigger_render)

	def _trigger_render(self):
		self._render_after = None
		if not self._busy:
			self._run_in_thread(self.render_from_cache)
		else:
			self._render_pending = True

	def _schedule_rebuild_l0(self):
		if self._rebuild_l0_after is not None:
			self.root.after_cancel(self._rebuild_l0_after)
		self._rebuild_l0_after = self.root.after(150, self._trigger_rebuild_l0)

	def _trigger_rebuild_l0(self):
		self._rebuild_l0_after = None
		def work():
			w, h = self.get_render_size()
			sc = self._get_sphere_coords(w, h) if self._cubemap_mode() else None
			if self._cubemap_mode():
				self.cached_l0 = make_L0_sphere(w, h, np.random.default_rng(self.l0_seed),
												self.l0_density.get(), self.l0_scale.get(), sc)
			else:
				self.cached_l0 = make_L0(w, h, np.random.default_rng(self.l0_seed),
										 self.l0_density.get(), self.l0_scale.get())
			self.render_from_cache()
		self._run_in_thread(work)

	def _schedule_rebuild_l1(self):
		if self._rebuild_l1_after is not None:
			self.root.after_cancel(self._rebuild_l1_after)
		self._rebuild_l1_after = self.root.after(150, self._trigger_rebuild_l1)

	def _trigger_rebuild_l1(self):
		self._rebuild_l1_after = None
		def work():
			w, h = self.get_render_size()
			if self._cubemap_mode():
				sc = self._get_sphere_coords(w, h)
				self.cached_l1 = make_L1_sphere(w, h, np.random.default_rng(self.l1_seed),
												self.l1_density.get(), self.l1_scale.get(), sc)
			else:
				self.cached_l1 = make_L1(w, h, np.random.default_rng(self.l1_seed),
										 self.l1_density.get(), self.l1_scale.get())
			self.render_from_cache()
		self._run_in_thread(work)

	def _generate_and_render_all(self):
		w, h = self.get_render_size()
		rng	 = np.random.default_rng(self.seed)
		self.l0_seed = self.seed
		self.l1_seed = self.seed + 1

		cubemap = self._cubemap_mode()
		if cubemap:
			sc = self._get_sphere_coords(w, h)
			self.cached_l0 = make_L0_sphere(w, h, np.random.default_rng(self.l0_seed),
											self.l0_density.get(), self.l0_scale.get(), sc)
			self.cached_l1 = make_L1_sphere(w, h, np.random.default_rng(self.l1_seed),
											self.l1_density.get(), self.l1_scale.get(), sc)
			self.cached_l2 = make_L2_sphere(w, h, np.random.default_rng(self.seed + 2),
											sphere_coords=sc)
			self.cached_l3 = make_L3_sphere(w, h, np.random.default_rng(self.seed + 3),
											sphere_coords=sc)
		else:
			self.cached_l0 = make_L0(w, h, np.random.default_rng(self.l0_seed),
									 self.l0_density.get(), self.l0_scale.get())
			self.cached_l1 = make_L1(w, h, np.random.default_rng(self.l1_seed),
									 self.l1_density.get(), self.l1_scale.get())
			self.cached_l2 = make_L2(w, h, np.random.default_rng(self.seed + 2))
			self.cached_l3 = make_L3(w, h, np.random.default_rng(self.seed + 3))

		self.render_from_cache()

	def generate_cache(self):
		self._generate_and_render_all()

	def regen_l0(self):
		self.l0_seed = random.randint(0, 10**12)
		w, h = self.get_render_size()
		seed = self.l0_seed
		cubemap = self._cubemap_mode()
		def work():
			if cubemap:
				sc = self._get_sphere_coords(w, h)
				self.cached_l0 = make_L0_sphere(w, h, np.random.default_rng(seed),
												self.l0_density.get(), self.l0_scale.get(), sc)
			else:
				self.cached_l0 = make_L0(w, h, np.random.default_rng(seed),
										 self.l0_density.get(), self.l0_scale.get())
			self.render_from_cache()
		self._run_in_thread(work)

	def regen_l1(self):
		self.l1_seed = random.randint(0, 10**12)
		w, h = self.get_render_size()
		seed = self.l1_seed
		cubemap = self._cubemap_mode()
		def work():
			if cubemap:
				sc = self._get_sphere_coords(w, h)
				self.cached_l1 = make_L1_sphere(w, h, np.random.default_rng(seed),
												self.l1_density.get(), self.l1_scale.get(), sc)
			else:
				self.cached_l1 = make_L1(w, h, np.random.default_rng(seed),
										 self.l1_density.get(), self.l1_scale.get())
			self.render_from_cache()
		self._run_in_thread(work)

	def regen_l3(self):
		w, h = self.get_render_size()
		cubemap = self._cubemap_mode()
		def work():
			s = random.randint(0, 10**12)
			if cubemap:
				sc = self._get_sphere_coords(w, h)
				self.cached_l3 = make_L3_sphere(w, h, np.random.default_rng(s), sphere_coords=sc)
			else:
				self.cached_l3 = make_L3(w, h, np.random.default_rng(s))
			self.render_from_cache()
		self._run_in_thread(work)

	def regen_l2(self):
		w, h   = self.get_render_size()
		custom = self.get_custom_colors()
		make_custom = [tuple((c * 255).astype(int)) for c in custom] if custom else None
		cubemap = self._cubemap_mode()
		def work():
			s = random.randint(0, 10**12)
			if cubemap:
				sc = self._get_sphere_coords(w, h)
				self.cached_l2 = make_L2_sphere(w, h, np.random.default_rng(s),
												custom_colors=make_custom, sphere_coords=sc)
			else:
				self.cached_l2 = make_L2(w, h, np.random.default_rng(s), custom_colors=make_custom)
			self.render_from_cache()
		self._run_in_thread(work)

	def render_from_cache(self):
		if self.cached_l0 is None:
			return

		w, h = self.get_render_size()
		l0 = self.cached_l0 * self.fine_star.get()
		l1 = self.cached_l1 * self.coarse_star.get()
		stars = np.clip((l0 + l1) * self.star_brightness.get(), 0.0, 1.0)
		after_wash = np.clip(stars + self.cached_l2 * self.wash_strength.get(), 0.0, 1.0)
		l3 = np.clip((self.cached_l3 - 0.5) * self.cloud_contrast.get() + 0.5, 0.0, 1.0)
		l3 = np.clip(l3 * self.cloud_strength.get(), 0.0, 1.0)
		result = blend_color_dodge(after_wash, l3)
		result = np.clip(result * self.brightness.get(), 0.0, 1.0)

		# im too lazy to make the vignette work on cubemap mode...
		vig = self.vignette.get()
		if vig > 0.0 and not self._cubemap_mode():
			yy, xx = np.mgrid[0:h, 0:w]
			dist = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
			result *= np.clip(1.0 - dist * vig, 0.0, 1.0)[..., None]

		pixel_data = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
		self.image	= pixel_data
		img	  = Image.fromarray(pixel_data, "RGB").resize((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
		photo = ImageTk.PhotoImage(img)
		def _update():
			self.photo = photo
			self.canvas.config(image=self.photo, bg="#1e1f22")
			if self._render_pending:
				self._render_pending = False
				self._schedule_render()
		self.root.after(0, _update)

	def get_custom_colors(self):
		text = self.custom_colors_entry.get().strip()
		if not text:
			return None
		try:
			colors = []
			for part in text.split(","):
				hx = part.strip().lstrip("#")
				if len(hx) != 6:
					continue
				r = int(hx[0:2], 16) / 255.0
				g = int(hx[2:4], 16) / 255.0
				b = int(hx[4:6], 16) / 255.0
				colors.append(np.array([r, g, b]))
			return colors if colors else None
		except Exception:
			return None

	def new_seed(self):
		self.seed = random.randint(0, 10**12)
		self.seed_var.set(str(self.seed))
		self.generate_cache()
		self.render_from_cache()

	def save_image(self):
		path = filedialog.asksaveasfilename(defaultextension=".png")
		if path and self.image is not None:
			Image.fromarray(self.image).save(path)

if __name__ == "__main__":
	root = tk.Tk()
	app	 = SpaceGUI(root)
	root.mainloop()