import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Polygon, Circle, Ellipse, Rectangle, FancyBboxPatch, PathPatch
)
from matplotlib.path import Path
import matplotlib.colors as mcolors
from io import BytesIO
from PIL import Image


COLOR_MAP = {
    "red":    (0.89, 0.22, 0.22),
    "blue":   (0.22, 0.47, 0.89),
    "green":  (0.22, 0.72, 0.35),
    "yellow": (0.95, 0.80, 0.15),
    "purple": (0.65, 0.35, 0.78),
    "orange": (0.95, 0.55, 0.15),
}

SIZE_SCALE = {
    "small":  0.55,
    "medium": 0.75,
    "large":  1.0,
}


def _apply_material(ax, shape, material, base_color, is_3d=False):
    """Apply material effects to a shape patch."""
    if material == "matte":
        shape.set_facecolor(base_color)
        shape.set_edgecolor("none")
        shape.set_alpha(1.0)

    elif material == "shiny":
        shape.set_facecolor(base_color)
        shape.set_edgecolor("none")
        rgb = mcolors.to_rgb(base_color)
        highlight = tuple(min(1.0, c + 0.35) for c in rgb)
        gradient = np.linspace(0, 1, 256).reshape(-1, 1) * np.array(highlight) + \
                   np.linspace(1, 0, 256).reshape(-1, 1) * np.array(rgb)
        ax.imshow(gradient.reshape(1, 256, 3), aspect='auto',
                  extent=shape.get_extent().extents if hasattr(shape, 'get_extent') else None,
                  alpha=0.3, zorder=shape.get_zorder() + 0.1)

    elif material == "metallic":
        rgb = mcolors.to_rgb(base_color)
        light = tuple(min(1.0, c * 1.6) for c in rgb)
        dark = tuple(c * 0.4 for c in rgb)
        shape.set_facecolor("none")
        try:
            extent = shape.get_extent()
            bbox = extent.extents if hasattr(extent, 'extents') else extent
        except (AttributeError, TypeError):
            bbox = None
        if bbox is None:
            shape.set_facecolor(base_color)
        else:
            x1, y1, x2, y2 = bbox
            grad = np.linspace(0, 1, 256)
            grad_img = np.zeros((256, 1, 3))
            for c in range(3):
                grad_img[:, 0, c] = grad * light[c] + (1 - grad) * dark[c]
            ax.imshow(grad_img, extent=[x1, x2, y1, y2],
                      aspect='auto', alpha=0.7, zorder=shape.get_zorder())
        shape.set_edgecolor(tuple(c * 0.25 for c in rgb))
        shape.set_linewidth(1.0)

    elif material == "glass":
        rgb = mcolors.to_rgb(base_color)
        shape.set_facecolor(base_color)
        shape.set_alpha(0.45)
        shape.set_edgecolor(tuple(c * 0.5 for c in rgb))
        shape.set_linewidth(1.5)

    return shape


def _draw_cube(ax, center, scale, color, material):
    """Draw a 3D-looking cube using polygons."""
    cx, cy = center
    s = 0.35 * scale
    dx, dy = 0.12 * scale, 0.12 * scale

    front = [(cx - s, cy - s), (cx + s, cy - s),
             (cx + s, cy + s), (cx - s, cy + s)]
    top = [(cx - s, cy - s), (cx - s + dx, cy - s - dy),
           (cx + s + dx, cy - s - dy), (cx + s, cy - s)]
    right = [(cx + s, cy - s), (cx + s + dx, cy - s - dy),
             (cx + s + dx, cy + s - dy), (cx + s, cy + s)]

    front_poly = Polygon(front, zorder=2)
    top_poly = Polygon(top, zorder=0)
    right_poly = Polygon(right, zorder=1)

    front_poly = _apply_material(ax, front_poly, material, color)
    top_poly.set_facecolor(tuple(min(1.0, c + 0.15) for c in mcolors.to_rgb(color)))
    right_poly.set_facecolor(tuple(max(0.0, c - 0.15) for c in mcolors.to_rgb(color)))

    if material == "glass":
        top_poly.set_alpha(0.35)
        right_poly.set_alpha(0.35)
    elif material == "metallic":
        top_poly.set_edgecolor("none")
        right_poly.set_edgecolor("none")

    ax.add_patch(front_poly)
    ax.add_patch(top_poly)
    ax.add_patch(right_poly)


def _draw_sphere(ax, center, scale, color, material):
    """Draw a sphere with radial gradient for 3D appearance."""
    cx, cy = center
    r = 0.38 * scale

    circle = Circle((cx, cy), r, zorder=3)
    circle = _apply_material(ax, circle, color, material)

    # Radial gradient for 3D shading
    if material != "glass":
        rgb = mcolors.to_rgb(color)
        n = 100
        for i in range(n, 0, -1):
            ri = r * i / n
            factor = 0.25 * (1 - i / n)
            shade = tuple(min(1.0, c + factor) for c in rgb)
            inner = Circle((cx, cy), ri, color=shade, zorder=3)
            ax.add_patch(inner)

    ax.add_patch(circle)


def _draw_cylinder(ax, center, scale, color, material):
    """Draw a cylinder (rectangle + ellipse top)."""
    cx, cy = center
    w = 0.25 * scale
    h = 0.35 * scale

    body = Rectangle((cx - w, cy - h), 2 * w, 2 * h, zorder=2)
    body = _apply_material(ax, body, color, material)

    top_ellipse = Ellipse((cx, cy - h), 2 * w, 0.08 * scale,
                          zorder=3)
    top_ellipse.set_facecolor(tuple(min(1.0, c + 0.2) for c in mcolors.to_rgb(color)))
    if material == "glass":
        top_ellipse.set_alpha(0.4)
        top_ellipse.set_edgecolor(tuple(c * 0.5 for c in mcolors.to_rgb(color)))
        top_ellipse.set_linewidth(1.0)

    bottom_ellipse = Ellipse((cx, cy + h), 2 * w, 0.06 * scale,
                             zorder=1)
    bottom_ellipse.set_facecolor(tuple(max(0.0, c - 0.15) for c in mcolors.to_rgb(color)))
    if material == "glass":
        bottom_ellipse.set_alpha(0.3)

    ax.add_patch(body)
    ax.add_patch(top_ellipse)
    ax.add_patch(bottom_ellipse)


def _draw_cone(ax, center, scale, color, material):
    """Draw a cone (triangle)."""
    cx, cy = center
    w = 0.32 * scale
    h = 0.40 * scale

    triangle = [(cx - w, cy + h), (cx + w, cy + h), (cx, cy - h)]
    cone = Polygon(triangle, zorder=2)
    cone = _apply_material(ax, cone, color, material)

    base = Ellipse((cx, cy + h), 2 * w, 0.08 * scale, zorder=1)
    base.set_facecolor(tuple(max(0.0, c - 0.2) for c in mcolors.to_rgb(color)))
    if material == "glass":
        base.set_alpha(0.3)

    ax.add_patch(cone)
    ax.add_patch(base)


def _draw_pyramid(ax, center, scale, color, material):
    """Draw a pyramid (triangle with different proportions and face line)."""
    cx, cy = center
    w = 0.35 * scale
    h = 0.35 * scale

    main = [(cx - w, cy + h), (cx + w, cy + h), (cx, cy - h)]
    pyramid = Polygon(main, zorder=2)
    pyramid = _apply_material(ax, pyramid, color, material)

    # Center line for pyramid distinction from cone
    ax.plot([cx, cx], [cy - h, cy + h], color='black', alpha=0.15,
            linewidth=1.0, zorder=3)

    ax.add_patch(pyramid)


SHAPE_DRAWERS = {
    "cube":     _draw_cube,
    "sphere":   _draw_sphere,
    "cylinder": _draw_cylinder,
    "cone":     _draw_cone,
    "pyramid":  _draw_pyramid,
}


class ShapeRenderer:
    """Programmatic renderer for 2D geometric shapes with attributes."""

    def __init__(self, image_size=224, bg_color=(0.96, 0.96, 0.96)):
        self.image_size = image_size
        self.bg_color = bg_color

    def render_single_object(self, obj_type, color_name=None, size="medium",
                             material="matte", angle_variant=0,
                             color_rgb_override=None):
        """Render a single object centered in the frame.

        Args:
            obj_type: str, one of cube, sphere, cylinder, cone, pyramid.
            color_name: str or None, named color from COLOR_MAP.
            size: str, one of small, medium, large.
            material: str, one of matte, shiny, metallic, glass.
            angle_variant: int, 0 for centered, ±1 for slight offset.
            color_rgb_override: tuple or None, direct (r,g,b) in [0,1].

        Returns:
            PIL.Image of the rendered object.
        """
        scale = SIZE_SCALE[size]
        if color_rgb_override is not None:
            color_rgb = color_rgb_override
        else:
            color_rgb = COLOR_MAP[color_name]
        angle_offset = (angle_variant - 1) * 0.05  # subtle variation

        dpi = 100
        figsize = self.image_size / dpi
        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        draw_fn = SHAPE_DRAWERS[obj_type]
        draw_fn(ax, (0.0 + angle_offset, 0.0), scale, color_rgb, material)

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    pad_inches=0, facecolor=self.bg_color)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return img

    def render_single_object_rgb(self, obj_type, color_rgb, size="medium",
                                  material="matte", angle_variant=0):
        """Render a single object with continuous RGB color.

        Args:
            obj_type: str, one of cube, sphere, cylinder, cone, pyramid.
            color_rgb: tuple, (r,g,b) in [0,1].
            size: str, one of small, medium, large.
            material: str, one of matte, shiny, metallic, glass.
            angle_variant: int, 0 for centered, ±1 for slight offset.

        Returns:
            PIL.Image of the rendered object.
        """
        return self.render_single_object(
            obj_type=obj_type, color_name=None, size=size,
            material=material, angle_variant=angle_variant,
            color_rgb_override=color_rgb)

    def render_scene(self, obj_a, obj_b, relation):
        """Render a dual-object scene.

        Args:
            obj_a: dict with keys obj_type, color, size, material
            obj_b: dict with keys obj_type, color, size, material
            relation: str, one of left_of, right_of, above, below, near, far

        Returns:
            PIL.Image of the rendered scene.
        """
        scale_a = SIZE_SCALE[obj_a["size"]]
        scale_b = SIZE_SCALE[obj_b["size"]]
        if "color_rgb" in obj_a:
            color_a = obj_a["color_rgb"]
        else:
            color_a = COLOR_MAP[obj_a["color"]]
        if "color_rgb" in obj_b:
            color_b = obj_b["color_rgb"]
        else:
            color_b = COLOR_MAP[obj_b["color"]]

        positions = _compute_positions(relation)

        dpi = 100
        figsize = self.image_size / dpi
        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        draw_fn_a = SHAPE_DRAWERS[obj_a["obj_type"]]
        draw_fn_b = SHAPE_DRAWERS[obj_b["obj_type"]]

        draw_fn_a(ax, positions["a"], scale_a, color_a, obj_a["material"])
        draw_fn_b(ax, positions["b"], scale_b, color_b, obj_b["material"])

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    pad_inches=0, facecolor=self.bg_color)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return img


    def render_scene_rgb(self, obj_a, obj_b, relation):
        """Render a scene where objects may have continuous RGB colors.

        Obj dicts may contain 'color_rgb' key with (r,g,b) tuple
        instead of 'color'. Delegates to render_scene which handles both.

        Returns:
            PIL.Image of the rendered scene.
        """
        return self.render_scene(obj_a, obj_b, relation)


def _compute_positions(relation):
    """Compute (x, y) positions for two objects based on relation."""
    if relation == "left_of":
        return {"a": (-0.30, 0.0), "b": (0.30, 0.0)}
    elif relation == "right_of":
        return {"a": (0.30, 0.0), "b": (-0.30, 0.0)}
    elif relation == "above":
        return {"a": (0.0, -0.25), "b": (0.0, 0.30)}
    elif relation == "below":
        return {"a": (0.0, 0.30), "b": (0.0, -0.25)}
    elif relation == "near":
        return {"a": (-0.20, 0.0), "b": (0.20, 0.0)}
    elif relation == "far":
        return {"a": (-0.42, 0.0), "b": (0.42, 0.0)}
    else:
        raise ValueError(f"Unknown relation: {relation}")
