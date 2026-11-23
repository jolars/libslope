#!/usr/bin/env python3
"""Generate the SLOPE logo as an SVG, PNG (via ImageMagick), or 3D OBJ."""

import argparse
import math
import re
import subprocess
from pathlib import Path

WIDTH, HEIGHT = 400, 425
HEIGHT_3D = 1.6
CAMERA_DIRECTION = (0.42, -0.94, 0.5)
CAMERA_DISTANCE = 9
FACE_COLOR = "222436"
OUTLINE_COLOR = "fefefe"
OUTLINE_WIDTH = 6


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    x, y, z = a
    u, v, w = b
    return (y * w - z * v, z * u - x * w, x * v - y * u)


def unit(vector):
    length = math.sqrt(dot(vector, vector))
    return tuple(value / length for value in vector)


def section(ratio, height):
    # For lambda = (a, b), the unit ball has |x|, |y| <= a
    # and |x| + |y| <= a + b. Keep its area constant along the height.
    a = 1 / math.sqrt(1 + 2 * ratio - ratio**2)
    b = ratio * a
    return [
        (x, y, height)
        for x, y in [
            (a, b),
            (b, a),
            (-b, a),
            (-a, b),
            (-a, -b),
            (-b, -a),
            (b, -a),
            (a, -b),
        ]
    ]


def logo_faces(bands):
    rings = [section(i / bands, HEIGHT_3D * i / bands) for i in range(bands + 1)]
    faces = [rings[0][::-1], rings[-1]]
    for lower, upper in zip(rings, rings[1:]):
        for i in range(8):
            j = (i + 1) % 8
            faces.append([lower[i], lower[j], upper[j], upper[i]])

    return [list(dict.fromkeys(face)) for face in faces]


def render_svg(
    faces, camera_direction, camera_distance, face_color, outline_color, outline_width
):
    view = unit(camera_direction)
    base_rgb = tuple(int(face_color[i : i + 2], 16) for i in (0, 2, 4))
    right = unit(cross((0, 0, 1), view))
    up = cross(view, right)
    light = unit((-0.55, -0.8, 1.6))

    camera = (
        camera_distance * view[0],
        camera_distance * view[1],
        HEIGHT_3D / 2 + camera_distance * view[2],
    )

    def project(point):
        offset = tuple(p - c for p, c in zip(point, camera))
        depth = -dot(offset, view)
        return dot(offset, right) / depth, -dot(offset, up) / depth

    projected = {point: project(point) for face in faces for point in face}
    coordinates = projected.values()
    left, top = map(min, zip(*coordinates))
    right_edge, bottom = map(max, zip(*coordinates))
    scale = min((WIDTH - 12) / (right_edge - left), (HEIGHT - 12) / (bottom - top))
    center_x = (left + right_edge) / 2
    center_y = (top + bottom) / 2

    polygons = []
    for face in faces:
        edges = [cross(a, b) for a, b in zip(face, face[1:] + face[:1])]
        normal = unit(tuple(map(sum, zip(*edges))))
        toward_camera = tuple(c - p for c, p in zip(camera, face[0]))
        if dot(normal, toward_camera) <= 0:
            continue
        shade = 0.47 + 0.51 * max(0, dot(normal, light))
        color = "#{:02x}{:02x}{:02x}".format(
            *(round(value * shade) for value in base_rgb)
        )
        points = " ".join(
            f"{WIDTH / 2 + scale * (x - center_x):.3f},"
            f"{HEIGHT / 2 + scale * (y - center_y):.3f}"
            for x, y in (projected[point] for point in face)
        )
        depth = sum(dot(point, view) for point in face) / len(face)
        polygons.append((depth, f'  <polygon fill="{color}" points="{points}"/>'))

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">',
            f'  <g stroke="#{outline_color}" stroke-width="{outline_width}" stroke-linejoin="round">',
            *(polygon for _, polygon in sorted(polygons)),
            "  </g>",
            "</svg>",
            "",
        ]
    )


def render_obj(faces):
    vertices = dict.fromkeys(point for face in faces for point in face)
    indices = {point: index for index, point in enumerate(vertices, start=1)}
    lines = ["o SLOPE"]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)
    lines.extend(
        "f " + " ".join(str(indices[point]) for point in face) for face in faces
    )
    return "\n".join(lines) + "\n"


def hex_color(value):
    color = value.removeprefix("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", color):
        raise argparse.ArgumentTypeError("expected a six-digit RGB hex color")
    return color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("slope-logo.png"),
        help="destination .svg, .png, or .obj (default: assets/slope-logo.png)",
    )
    parser.add_argument("--bands", type=int, default=4, help="number of vertical bands")
    parser.add_argument(
        "--camera",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=CAMERA_DIRECTION,
        help="direction from the model to the camera (default: 0.48 -0.84 0.34)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=CAMERA_DISTANCE,
        help="camera distance from the model center (default: 6)",
    )
    parser.add_argument(
        "--color",
        type=hex_color,
        default=FACE_COLOR,
        help="face color as an RGB hex value (default: 9eb7b1)",
    )
    parser.add_argument(
        "--outline",
        type=hex_color,
        default=OUTLINE_COLOR,
        help="outline color as an RGB hex value (default: 172522)",
    )
    parser.add_argument(
        "--outline-width",
        type=float,
        default=OUTLINE_WIDTH,
        help=f"outline width in pixels (default: {OUTLINE_WIDTH}, 0 hides outlines)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.outline_width) or args.outline_width < 0:
        parser.error("--outline-width must be finite and nonnegative")
    if args.bands < 2:
        parser.error("--bands must be at least 2")
    suffix = args.output.suffix.lower()
    if suffix not in {".svg", ".png", ".obj"}:
        parser.error("output must end in .svg, .png, or .obj")

    if not all(map(math.isfinite, args.camera)) or not any(args.camera[:2]):
        parser.error("--camera needs finite coordinates and a nonzero X or Y")
    if not math.isfinite(args.distance) or args.distance < 2:
        parser.error("--distance must be finite and at least 2")

    faces = logo_faces(args.bands)
    if suffix == ".obj":
        args.output.write_text(render_obj(faces), encoding="utf-8")
        return

    svg = render_svg(
        faces, args.camera, args.distance, args.color, args.outline, args.outline_width
    )
    if suffix == ".svg":
        args.output.write_text(svg, encoding="utf-8")
    else:
        try:
            subprocess.run(
                [
                    "magick",
                    "-background",
                    "none",
                    "-density",
                    "384",
                    "MSVG:-",
                    "-resize",
                    f"{WIDTH}x{HEIGHT}",
                    "-depth",
                    "8",
                    "-strip",
                    f"PNG32:{args.output}",
                ],
                input=svg.encode("utf-8"),
                check=True,
            )
        except FileNotFoundError:
            parser.exit(1, "PNG export requires ImageMagick (magick).\n")


if __name__ == "__main__":
    main()
