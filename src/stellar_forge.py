import argparse, datetime, random, sys
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageTk
import numpy as np
from numba import njit, prange
import tkinter as tk
from tkinter import ttk, filedialog
import ctypes

PREVIEW_W = 1280
PREVIEW_H = 720

def blend_color_dodge(bottom, top):
	if top.ndim == 2:
		top = top[..., None]
	if top.shape[-1] == 1 and bottom.shape[-1] == 3:
		top = np.repeat(top, 3, axis=2)
	denom = np.clip(1.0 - top, 1e-6, 1.0)
	return np.clip(bottom / denom, 0.0, 1.0)

def apply_levels(arr, in_black, gamma, in_white):
	lo = in_black / 255.0
	hi = in_white / 255.0
	arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
	return np.power(arr, 1.0 / gamma).astype(np.float32)

def add_noise(width, height, rng, std):
	noise = np.abs(rng.normal(0.0, std, (height, width)).astype(np.float32))
	return np.clip(noise, 0.0, 1.0)

def make_L0(width, height, rng, density=0.5, scale=0.5):
	noise = add_noise(width, height, rng, density)
	img = Image.fromarray((noise * 255).astype(np.uint8), "L")
	img = img.filter(ImageFilter.GaussianBlur(radius=scale))
	noise = np.array(img).astype(np.float32) / 255.0
	noise = apply_levels(noise, in_black=170, gamma=0.40, in_white=255)
	return np.stack([noise, noise, noise], axis=-1)

def make_L1(width, height, rng, density=0.6, scale=1.5):
	noise = add_noise(width, height, rng, density)
	img = Image.fromarray((noise * 255).astype(np.uint8), "L")
	img = img.filter(ImageFilter.GaussianBlur(radius=scale))
	noise = np.array(img).astype(np.float32) / 255.0
	noise = apply_levels(noise, in_black=170, gamma=1.0, in_white=255)
	return np.stack([noise, noise, noise], axis=-1)

# early version template. decided to leave it.
palettes = { 
		"warm":	 [(255, 80,	 30),  (255, 160, 20),	(200, 40,  80),	 (255, 200, 80)],
		"cool":	 [(30,	80,	 255), (20,	 200, 255), (80,  30,  200), (40,  180, 220)],
		"mixed": [(180, 20,	 255), (20,	 220, 180), (255, 60,  120), (60,  120, 255)],
	}
	
def rand_color(rng, palette):
	base = palettes[palette][rng.integers(0, len(palettes[palette]))]
	return tuple(max(0, min(255, c + int(rng.integers(-30, 31)))) for c in base)

def make_L2(width, height, rng, custom_colors=None):
	if custom_colors:
		active_palette = custom_colors
	else:
		key = rng.choice(["warm", "cool", "mixed"])
		active_palette = palettes[key]

	REF_W = 800.0
	REF_H = 450.0
	scale_x = width / REF_W
	scale_y = height / REF_H
	scale = min(scale_x, scale_y)
	num_blobs = int(rng.integers(3, 8))
	ref_blur = min(REF_W, REF_H) / 5.0
	blur_radius = int(ref_blur * scale)
	img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8), "RGB")
	draw = ImageDraw.Draw(img)
	
	for _ in range(num_blobs):
		base = active_palette[rng.integers(0, len(active_palette))]
		color = tuple(max(0, min(255, c + int(rng.integers(-30, 31)))) for c in base)
		cx = int(rng.uniform(0, REF_W) * scale_x)
		cy = int(rng.uniform(0, REF_H) * scale_y)
		ref_rx = rng.uniform(REF_W / 6.0, REF_W / 2.0)
		ref_ry = rng.uniform(REF_H / 6.0, REF_H / 2.0)
		rx = int(ref_rx * scale_x)
		ry = int(ref_ry * scale_y)
		draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
		
	img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
	layer = np.array(img).astype(np.float32) / 255.0
	layer *= 0.30
	return layer

# not my code and i dont want to comment anything about it
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
	xi = int(np.floor(x))
	yi = int(np.floor(y))
	xf = x - xi
	yf = y - yi 
	u = _fade(xf)
	v = _fade(yf)
	aa = _hash2d(xi,	 yi,	 seed)
	ba = _hash2d(xi + 1, yi,	 seed)
	ab = _hash2d(xi,	 yi + 1, seed)
	bb = _hash2d(xi + 1, yi + 1, seed)
	g_aa = _grad(aa, xf,	 yf)
	g_ba = _grad(ba, xf - 1, yf)
	g_ab = _grad(ab, xf,	 yf - 1)
	g_bb = _grad(bb, xf - 1, yf - 1)
	x1 = _lerp(g_aa, g_ba, u)
	x2 = _lerp(g_ab, g_bb, u)
	return _lerp(x1, x2, v)

@njit(cache=True)
def _perlin2d_octaves(x, y, octaves, seed):
	total = 0.0
	amplitude = 1.0
	frequency = 1.0
	max_amp = 0.0
	
	for _ in range(octaves):
		total += amplitude * _perlin2d(x * frequency, y * frequency, seed)
		max_amp += amplitude
		amplitude *= 0.5
		frequency *= 2.0  
	return total / max_amp

@njit(parallel=True, cache=True)
def _perlin_array(xv, yv, octaves, seed):
	h, w = xv.shape
	out = np.empty((h, w), dtype=np.float32)
	for i in prange(h):
		for j in range(w):
			out[i, j] = _perlin2d_octaves(xv[i, j], yv[i, j], octaves, seed)
	return out

def perlin_field(width, height, scale, octaves, ox, oy, seed=0):
	ref_w = 800.0
	ref_h = 450.0
	x = np.linspace(ox, ox + ref_w / scale, width, dtype=np.float64)
	y = np.linspace(oy, oy + ref_h / scale, height, dtype=np.float64)
	xv, yv = np.meshgrid(x, y)
	arr = _perlin_array(xv, yv, octaves, seed)
	arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
	return arr

def make_L3(width, height, rng):
	scale	= rng.uniform(250, 450)
	octaves = 6
	ox1, oy1 = float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))
	ox2, oy2 = float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))

	p1 = perlin_field(width, height, scale, octaves, ox1, oy1, seed=42)
	p2 = perlin_field(width, height, scale, octaves, ox2, oy2, seed=137)

	base = np.abs(np.abs(0.5 - p1) - p2)
	base = (base - base.min()) / (base.max() - base.min() + 1e-6)
	return base

def composite(width, height, rng):
	l0 = make_L0(width, height, rng)
	l1 = make_L1(width, height, rng)
	star_base = np.clip(l0 + l1, 0.0, 1.0)
	l2 = make_L2(width, height, rng)
	after_wash = np.clip(star_base + l2 * 0.5, 0.0, 1.0)
	l3 = make_L3(width, height, rng)
	result = blend_color_dodge(after_wash, l3)
	result = np.clip(result, 0.0, 1.0)
	yy, xx = np.mgrid[0:height, 0:width]
	dist = np.sqrt(((xx - width/2) / (width/2))**2 +
				   ((yy - height/2) / (height/2))**2)
	result *= np.clip(1.0 - dist * 0.4, 0.0, 1.0)[..., None]
	return (result * 255).astype(np.uint8)

class SpaceGUI:
	def slider(self, parent, label, var, frm, to):
		row = ttk.Frame(parent)
		row.pack(fill="x", pady=2)
		top = ttk.Frame(row)
		top.pack(fill="x")
		ttk.Label(top, text=label).pack(side="left")
		value_label = ttk.Label(
			top,
			text=f"{var.get():.2f}",
			foreground="#00e5ff"
		)
		value_label.pack(side="right")
		def update(val):
			value_label.config(text=f"{float(val):.2f}")
			self.render_from_cache()
		ttk.Scale(
			row,
			from_=frm,
			to=to,
			variable=var,
			style="Red.Horizontal.TScale",
			command=update
		).pack(fill="x", expand=True)

    # this is not ideal but i got a headache trying to fix the borderless window work. might rework later
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
		WS_CAPTION = 0x00C00000
		WS_THICKFRAME = 0x00040000
		WS_MINIMIZEBOX = 0x00020000
		WS_MAXIMIZEBOX = 0x00010000

		style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
		style = style & ~WS_CAPTION & ~WS_THICKFRAME
		ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, style)

		ctypes.windll.user32.SetWindowPos(
			hwnd, None, 0, 0, 0, 0,
			0x0020 | 0x0001 | 0x0002
		)

	def __init__(self, root):
		self.root = root
		self.root.title("Stellar Forge by Madness (null138)")
		self.border_width = 5  
		self.bg_canvas = tk.Canvas(root, highlightthickness=0, bd=0, width=1600, height=920)
		self.bg_canvas.pack(fill="both", expand=True)

		self.main_ui = tk.Frame(self.bg_canvas, bg="#1e1f22")
		self.main_ui.place(relx=0, rely=0, relwidth=1, relheight=1, 
						   x=self.border_width, y=self.border_width, 
						   width=-self.border_width*2, height=-self.border_width*2)

		def draw_gradient(event=None):
			w = self.bg_canvas.winfo_width()
			h = self.bg_canvas.winfo_height()
			self.bg_canvas.delete("grad")
			steps = 64 
			for i in range(steps):
				r = int(255 - (255 * i / steps))
				g = int(0 + (229 * i / steps))
				b = 255
				color = f'#{r:02x}{g:02x}{b:02x}'
				self.bg_canvas.create_rectangle(
					0, 0, w * (1 - i/steps), h * (1 - i/steps),
					fill=color, outline=color, tags="grad"
				)
			self.bg_canvas.tag_lower("grad")
		self.bg_canvas.bind("<Configure>", draw_gradient)

		self.seed = random.randint(0, 10**12)
		self.cached_l0 = None
		self.cached_l1 = None
		self.cached_l2 = None
		self.cached_l3 = None

		self.fine_star = tk.DoubleVar(value=1.0)
		self.coarse_star = tk.DoubleVar(value=1.0)
		self.star_brightness = tk.DoubleVar(value=1.0)
		self.cloud_strength = tk.DoubleVar(value=1.0)
		self.cloud_contrast = tk.DoubleVar(value=1.0)
		self.wash_strength = tk.DoubleVar(value=1.0)
		self.vignette = tk.DoubleVar(value=0.0)
		self.brightness = tk.DoubleVar(value=1.0)
		self.l0_density = tk.DoubleVar(value=0.5)
		self.l0_scale	= tk.DoubleVar(value=0.5)

		self.l1_density = tk.DoubleVar(value=0.6)
		self.l1_scale	= tk.DoubleVar(value=1.5)

		self.res_width_var	= tk.StringVar(value="1920")
		self.res_height_var = tk.StringVar(value="1080")

		titlebar = tk.Frame(self.main_ui, bg="#2a2d31", height=32)
		titlebar.pack(side="top", fill="x")
		titlebar.pack_propagate(False)

		tk.Label(titlebar, text="Stellar Forge by Madness (null138)", bg="#2a2d31", fg="#00e5ff",
				 font=("Segoe UI", 10)).pack(side="left", padx=10)

		tk.Button(titlebar, text="✕", bg="#2a2d31", fg="#e6e6e6", bd=0,
				  activebackground="#e81123", activeforeground="white", width=4,
				  command=root.destroy).pack(side="right")

		tk.Button(titlebar, text="─", bg="#2a2d31", fg="#e6e6e6", bd=0,
				  activebackground="#3a3d42", activeforeground="white", width=4,
				  command=self._minimize).pack(side="right")

		titlebar.bind("<ButtonPress-1>", self._drag_start)
		titlebar.bind("<B1-Motion>", self._drag_move)

		controls_border = tk.Frame(self.main_ui, bg="#3a3d42", bd=0, relief="flat", padx=2, pady=2)
		controls_border.pack(side="left", fill="y", padx=10, pady=10)

		controls = ttk.Frame(controls_border, padding=10)
		controls.pack(fill="y", expand=True)

		style = ttk.Style()
		style.theme_use("clam")

		BG = "#1e1f22"
		PANEL = "#2a2d31"
		ACCENT = "#00e5ff"
		ACCENT_DARK = "#00a3b3"
		TEXT = "#e6e6e6"
		MUTED = "#a0a0a0"

		style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL)
		style.configure("TFrame", background=BG)
		style.configure("TLabel", background=BG, foreground=TEXT)
		style.configure("TLabelframe", background=BG, foreground=TEXT, borderwidth=0, relief="flat")
		style.configure("TLabelframe.Label", background=BG, foreground=ACCENT)
		style.configure("TButton",
			padding=6,
			background=PANEL,
			foreground=TEXT,
			borderwidth=0
		)
		style.map("TButton",
			background=[("active", ACCENT_DARK)],
			foreground=[("active", "#ffffff")]
		)
		style.configure("Red.Horizontal.TScale",
			background=BG,
			troughcolor=PANEL,
			sliderthickness=14,
			lightcolor="#ff4d4d",
			darkcolor="#b30000",
			bordercolor="#ff1a1a"
		)
		style.configure("TEntry",
			fieldbackground=PANEL,
			foreground=TEXT,
			insertcolor=ACCENT
		)

		stars_frame = ttk.LabelFrame(controls, text="Stars", padding=8)
		stars_frame.pack(fill="x", pady=5)
		stars_grid = ttk.Frame(stars_frame)
		stars_grid.pack(fill="x")
		fine_col = ttk.Frame(stars_grid)
		fine_col.grid(row=0, column=0, padx=6, sticky="nsew")
		self.slider(fine_col, "Fine", self.fine_star, 0.0, 3.0)
		ttk.Button(fine_col, text="🔄 Regenerate Fine", command=self.regen_l0).pack(fill="x", pady=2)
		self.slider(fine_col, "Fine Density", self.l0_density, 0.0, 1.0)
		self.slider(fine_col, "Fine Suppress", self.l0_scale, 0.0, 3.0)

		coarse_col = ttk.Frame(stars_grid)
		coarse_col.grid(row=0, column=1, padx=6, sticky="nsew")
		self.slider(coarse_col, "Coarse", self.coarse_star, 0.0, 3.5)
		ttk.Button(coarse_col, text="🔄 Regenerate Coarse", command=self.regen_l1).pack(fill="x", pady=2)
		self.slider(coarse_col, "Coarse Density", self.l1_density, 0.0, 1.0)
		self.slider(coarse_col, "Coarse Suppress", self.l1_scale, 0.0, 3.0)

		self.slider(stars_frame, "Brightness", self.star_brightness, 0.0, 2.0)

		clouds_frame = ttk.LabelFrame(controls, text="Clouds", padding=8)
		clouds_frame.pack(fill="x", pady=5)
		self.slider(clouds_frame, "Strength", self.cloud_strength, 0.0, 2.0)
		ttk.Button(clouds_frame, text="🔄 Regenerate Clouds", command=self.regen_l3).pack(fill="x", pady=2)
		self.slider(clouds_frame, "Contrast", self.cloud_contrast, 0.0, 3.0)

		color_frame = ttk.LabelFrame(controls, text="Color", padding=8)
		color_frame.pack(fill="x", pady=5)
		self.slider(color_frame, "Wash", self.wash_strength, 0.0, 2.0)
		ttk.Button(color_frame, text="🔄 Regenerate Color", command=self.regen_l2).pack(fill="x", pady=2)
		ttk.Label(color_frame, text="Custom colors (hex, comma):").pack(anchor="w")
		self.custom_colors_entry = ttk.Entry(color_frame)
		self.custom_colors_entry.pack(fill="x", pady=2)
		self.custom_colors_entry.insert(0, "")

		global_frame = ttk.LabelFrame(controls, text="Global", padding=8)
		global_frame.pack(fill="x", pady=5)
		self.slider(global_frame, "Vignette", self.vignette, 0.0, 2.0)
		self.slider(global_frame, "Brightness", self.brightness, 0.0, 3.0)

		res_frame = ttk.LabelFrame(controls, text="Resolution", padding=8)
		res_frame.pack(fill="x", pady=5)
		res_row = ttk.Frame(res_frame)
		res_row.pack(fill="x")
		ttk.Label(res_row, text="Width").pack(side="left")
		ttk.Entry(res_row, textvariable=self.res_width_var, width=6).pack(side="left", padx=(2, 6))
		ttk.Label(res_row, text="Height").pack(side="left")
		ttk.Entry(res_row, textvariable=self.res_height_var, width=6).pack(side="left", padx=(2, 0))
		ttk.Label(res_frame, text="(Takes effect after generating a new seed)",
				  foreground=MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

		btn_frame = ttk.Frame(controls)
		btn_frame.pack(fill="x", pady=10)
		ttk.Button(btn_frame, text="Generate New seed", command=self.new_seed).pack(fill="x", pady=2)
		ttk.Button(btn_frame, text="Save", command=self.save_image).pack(fill="x", pady=2)

		canvas_border = tk.Frame(self.main_ui, bg="#3a3d42", bd=2, relief="flat")
		canvas_border.pack(side="right", expand=True, padx=10, pady=10)
		self.canvas = tk.Label(canvas_border, bg="#1e1f22",
							   width=PREVIEW_W, height=PREVIEW_H)
		self.canvas.pack(padx=2, pady=2)
		self.image = None
		self.generate_cache()
		self.render_from_cache()
		self.root.after(500, self.make_borderless)

	def generate_cache(self):
		w, h = self.get_render_size()
		rng = np.random.default_rng(self.seed)
		self.l0_seed = self.seed
		self.l1_seed = self.seed + 1
		self.cached_l0 = make_L0(
			w, h, rng,
			self.l0_density.get(),
			self.l0_scale.get()
		)
		self.cached_l1 = make_L1(
			w, h, rng,
			self.l1_density.get(),
			self.l1_scale.get()
		)
		self.cached_l2 = make_L2(w, h, rng)
		self.cached_l3 = make_L3(w, h, rng)

	def regen_l0(self):
		self.l0_seed = random.randint(0, 10**12)
		w, h = self.get_render_size()
		rng = np.random.default_rng(self.l0_seed)
		self.cached_l0 = make_L0(
			w, h, rng,
			self.l0_density.get(),
			self.l0_scale.get()
		)
		self.render_from_cache()

	def regen_l1(self):
		self.l1_seed = random.randint(0, 10**12)
		w, h = self.get_render_size()
		rng = np.random.default_rng(self.l1_seed)
		self.cached_l1 = make_L1(
			w, h, rng,
			self.l1_density.get(),
			self.l1_scale.get()
		)
		self.render_from_cache()

	def regen_l3(self):
		w, h = self.get_render_size()
		rng = np.random.default_rng(random.randint(0, 10**12))
		self.cached_l3 = make_L3(w, h, rng)
		self.render_from_cache()

	def regen_l2(self):
		w, h = self.get_render_size()
		rng = np.random.default_rng(random.randint(0, 10**12))
		custom = self.get_custom_colors()
		make_custom = None
		if custom:
			make_custom = [tuple((c * 255).astype(int)) for c in custom]
		
		self.cached_l2 = make_L2(w, h, rng, custom_colors=make_custom)
		self.render_from_cache()

	def render_from_cache(self):
		w, h = self.get_render_size()
		rng0 = np.random.default_rng(self.l0_seed)
		rng1 = np.random.default_rng(self.l1_seed)
		l0 = make_L0(w, h, rng0, self.l0_density.get(), self.l0_scale.get())
		l1 = make_L1(w, h, rng1, self.l1_density.get(), self.l1_scale.get())
		l0 *= self.fine_star.get()
		l1 *= self.coarse_star.get()
		stars = np.clip((l0 + l1) * self.star_brightness.get(), 0, 1)
		wash = self.cached_l2
		after_wash = np.clip(stars + wash * self.wash_strength.get(), 0, 1)
		l3 = self.cached_l3.copy()
		l3 = np.clip((l3 - 0.5) * self.cloud_contrast.get() + 0.5, 0, 1)
		l3 = np.clip(l3 * self.cloud_strength.get(), 0, 1) 
		result = blend_color_dodge(after_wash, l3)
		result = np.clip(result * self.brightness.get(), 0, 1)	
		yy, xx = np.mgrid[0:h, 0:w]
		dist = np.sqrt(((xx - w/2) / (w/2))**2 + ((yy - h/2) / (h/2))**2)
		result *= np.clip(1.0 - dist * self.vignette.get(), 0, 1)[..., None]  
		self.image = (np.clip(result, 0, 1) * 255).astype(np.uint8)
		img = Image.fromarray(self.image, "RGB").resize((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
		self.photo = ImageTk.PhotoImage(img)
		self.canvas.config(image=self.photo, bg="#1e1f22")

	def get_custom_colors(self):
		text = self.custom_colors_entry.get().strip()
		if not text:
			return None
		try:
			colors = []
			for part in text.split(","):
				hex_color = part.strip().lstrip("#")
				if len(hex_color) != 6:
					continue
				r = int(hex_color[0:2], 16) / 255.0
				g = int(hex_color[2:4], 16) / 255.0
				b = int(hex_color[4:6], 16) / 255.0
				colors.append(np.array([r, g, b]))
			return colors if colors else None
		except:
			return None

	def new_seed(self):
		self.seed = random.randint(0, 10**12)
		self.generate_cache()
		self.render_from_cache()

	def save_image(self):
		path = filedialog.asksaveasfilename(defaultextension=".png")
		if path:
			Image.fromarray(self.image).save(path)

if __name__ == "__main__":
	root = tk.Tk()
	app = SpaceGUI(root)
	root.mainloop()