import torch
from typing import Dict, List, Tuple

from .config import CaseConfig, ShapeSpec
from .utils.math import getPeriodicPositions
from .utils.sdf import getSDF


def translate(points, offset):
    return points + offset


def rotate(points, angle):
    c = torch.cos(torch.tensor(angle, device=points.device, dtype=points.dtype))
    s = torch.sin(torch.tensor(angle, device=points.device, dtype=points.dtype))
    rotationMatrix = torch.tensor([[c, -s], [s, c]], device=points.device, dtype=points.dtype)
    return points @ rotationMatrix.T


def sample_sphere(
    particleState,
    config,
    radius: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    pre_rotation: float = 0.0,
    post_rotation: float = 0.0,
):
    positions = getPeriodicPositions(particleState.positions, config.domain)
    pre_rotated = rotate(positions, post_rotation)
    translated = translate(pre_rotated, torch.tensor([-offset[0], -offset[1]], device=positions.device))
    post_rotated = rotate(translated, pre_rotation)
    sdf = lambda points: getSDF("circle")["function"](points, radius)
    return torch.where(sdf(post_rotated) < 0, 1, 0).float()


def sample_box(
    particleState,
    config,
    half_extents: Tuple[float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
    pre_rotation: float = 0.0,
    post_rotation: float = 0.0,
):
    positions = getPeriodicPositions(particleState.positions, config.domain)
    pre_rotated = rotate(positions, post_rotation)
    translated = translate(pre_rotated, torch.tensor([-offset[0], -offset[1]], device=positions.device))
    post_rotated = rotate(translated, pre_rotation)
    sdf = lambda points: getSDF("box")["function"](points, torch.tensor(half_extents, device=points.device))
    return torch.where(sdf(post_rotated) < 0, 1, 0).float()


def sample_horizontal_line(
    particleState,
    config,
    thickness: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    pre_rotation: float = 0.0,
):
    length = (config.domain.max[0] - config.domain.min[0]).item() if torch.is_tensor(config.domain.max[0]) else (config.domain.max[0] - config.domain.min[0])
    return sample_box(
        particleState,
        config,
        half_extents=(length / 2, thickness / 2),
        offset=offset,
        pre_rotation=pre_rotation,
    )


def sample_vertical_line(
    particleState,
    config,
    thickness: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    pre_rotation: float = 0.0,
):
    length = (config.domain.max[1] - config.domain.min[1]).item() if torch.is_tensor(config.domain.max[1]) else (config.domain.max[1] - config.domain.min[1])
    return sample_box(
        particleState,
        config,
        half_extents=(thickness / 2, length / 2),
        offset=offset,
        pre_rotation=pre_rotation,
    )


def sample_vesica(
    particleState,
    config,
    radius: float,
    width: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    pre_rotation: float = 0.0,
):
    positions = getPeriodicPositions(particleState.positions, config.domain)
    pre_rotated = rotate(positions, 0.0)
    translated = translate(pre_rotated, torch.tensor([-offset[0], -offset[1]], device=positions.device))
    post_rotated = rotate(translated, pre_rotation)
    sdf = lambda points: getSDF("vesica")["function"](points, radius, width)
    return torch.where(sdf(post_rotated) < 0, 1, 0).float()


def sample_triangle(
    particleState,
    config,
    v0: Tuple[float, float],
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
    pre_rotation: float = 0.0,
):
    positions = getPeriodicPositions(particleState.positions, config.domain)
    translated = translate(positions, torch.tensor([-offset[0], -offset[1]], device=positions.device))
    post_rotated = rotate(translated, pre_rotation)
    sdf = lambda points: getSDF("triangle")["function"](
        points,
        torch.tensor(v0, device=points.device),
        torch.tensor(v1, device=points.device),
        torch.tensor(v2, device=points.device),
    )
    return torch.where(sdf(post_rotated) < 0, 1, 0).float()


def sample_equilateral_triangle(
    particleState,
    config,
    side_length: float,
    offset: Tuple[float, float] = (0.0, 0.0),
    pre_rotation: float = 0.0,
):
    height = (3**0.5) / 2 * side_length
    v0 = (-side_length / 2, -height / 3)
    v1 = (side_length / 2, -height / 3)
    v2 = (0.0, 2 * height / 3)
    return sample_triangle(particleState, config, v0, v1, v2, offset, pre_rotation)


def normalize_shape_kind(kind: str) -> str:
    aliases = {
        "circle": "sphere",
        "square": "box",
        "line": "horizontal_line",
        "equilateralTriangle": "equilateral_triangle",
    }
    return aliases.get(kind, kind)


def _sample_slit_obstacle(
    particleState,
    config,
    slot_widths: List[float],
    slot_heights: List[float],
    wall_thickness: float = 0.04,
    wall_offset: Tuple[float, float] = (0.0, 0.0),
    slit_x_half_extent: float = 0.05,
    pre_rotation: float = 0.0,
):
    wall = sample_vertical_line(
        particleState,
        config,
        thickness=wall_thickness,
        offset=wall_offset,
        pre_rotation=pre_rotation,
    )
    obstacle = wall.clone()
    for slot_width, slot_height in zip(slot_widths, slot_heights):
        slit = sample_box(
            particleState,
            config,
            half_extents=(slit_x_half_extent, slot_width / 2),
            offset=(wall_offset[0], wall_offset[1] + slot_height),
            pre_rotation=pre_rotation,
        )
        obstacle = obstacle - slit
    return torch.clamp(obstacle, min=0.0, max=1.0)


def sample_shape_structured(particleState, config, shape: ShapeSpec):
    kind = normalize_shape_kind(shape.kind)
    params = shape.params or {}
    offset = tuple(shape.position)
    rotation = float(shape.rotation)

    if kind == "sphere":
        return sample_sphere(particleState, config, radius=float(params.get("radius", 0.1)), offset=offset, pre_rotation=rotation)

    if kind == "box":
        if "half_extents" in params:
            half_extents = tuple(params["half_extents"])
        elif "size" in params:
            size = tuple(params["size"])
            half_extents = (float(size[0]) * 0.5, float(size[1]) * 0.5)
        else:
            radius = float(params.get("radius", 0.1))
            half_extents = (radius, radius)
        return sample_box(particleState, config, half_extents=half_extents, offset=offset, pre_rotation=rotation)

    if kind == "horizontal_line":
        return sample_horizontal_line(
            particleState,
            config,
            thickness=float(params.get("thickness", 0.04)),
            offset=offset,
            pre_rotation=rotation,
        )

    if kind == "vertical_line":
        return sample_vertical_line(
            particleState,
            config,
            thickness=float(params.get("thickness", 0.04)),
            offset=offset,
            pre_rotation=rotation,
        )

    if kind == "vesica":
        radius = float(params.get("radius", 0.1))
        return sample_vesica(
            particleState,
            config,
            radius=radius,
            width=float(params.get("width", radius * 0.5)),
            offset=offset,
            pre_rotation=rotation,
        )

    if kind == "triangle":
        vertices = params.get("vertices")
        if vertices is None:
            side_length = float(params.get("side_length", 0.2))
            height = (3**0.5) / 2 * side_length
            vertices = [(-side_length / 2, -height / 3), (side_length / 2, -height / 3), (0.0, 2 * height / 3)]
        return sample_triangle(
            particleState,
            config,
            v0=tuple(vertices[0]),
            v1=tuple(vertices[1]),
            v2=tuple(vertices[2]),
            offset=offset,
            pre_rotation=rotation,
        )

    if kind == "equilateral_triangle":
        return sample_equilateral_triangle(
            particleState,
            config,
            side_length=float(params.get("side_length", 0.2)),
            offset=offset,
            pre_rotation=rotation,
        )

    if kind == "single_slit":
        return _sample_slit_obstacle(
            particleState,
            config,
            slot_widths=[float(params.get("slot_width", 0.05))],
            slot_heights=[float(params.get("slot_height", 0.0))],
            wall_thickness=float(params.get("wall_thickness", 0.04)),
            wall_offset=offset,
            slit_x_half_extent=float(params.get("slit_x_half_extent", 0.05)),
            pre_rotation=rotation,
        )

    if kind == "double_slit":
        return _sample_slit_obstacle(
            particleState,
            config,
            slot_widths=list(params.get("slot_widths", [0.05, 0.05])),
            slot_heights=list(params.get("slot_heights", [0.2, -0.2])),
            wall_thickness=float(params.get("wall_thickness", 0.04)),
            wall_offset=offset,
            slit_x_half_extent=float(params.get("slit_x_half_extent", 0.05)),
            pre_rotation=rotation,
        )

    if kind == "prism":
        return sample_equilateral_triangle(
            particleState,
            config,
            side_length=float(params.get("side_length", 0.3)),
            offset=offset,
            pre_rotation=float(params.get("prism_rotation", rotation)),
        )

    raise ValueError(f"Unknown structured shape kind: {shape.kind}")


def _legacy_to_shape_spec(entity, default_kind: str) -> ShapeSpec:
    shape_spec = getattr(entity, "shapeSpec", None)
    if shape_spec is not None:
        return shape_spec

    shape = getattr(entity, "shape", default_kind)
    position = getattr(entity, "position", (0.0, 0.0))
    radius = float(getattr(entity, "radius", 0.1))
    size = getattr(entity, "size", None)

    kind = normalize_shape_kind(shape)
    params: Dict[str, object] = {}
    if kind in ("sphere", "vesica"):
        params["radius"] = radius
    elif kind == "box":
        params["half_extents"] = (radius, radius)
    elif kind == "horizontal_line":
        params["thickness"] = radius * 0.2
    elif kind == "equilateral_triangle":
        params["side_length"] = radius * 2
    if size is not None and kind == "box":
        params["size"] = tuple(size)
    if kind == "vesica":
        params["width"] = radius * 0.5

    return ShapeSpec(kind=kind, position=position, params=params)


def _default_obstacle_speed(case_config: CaseConfig, obstacle_index: int) -> float:
    if isinstance(case_config.defaultObstacleSpeed, list):
        if len(case_config.defaultObstacleSpeed) == 0:
            return 0.5
        return float(case_config.defaultObstacleSpeed[min(obstacle_index, len(case_config.defaultObstacleSpeed) - 1)])
    return float(case_config.defaultObstacleSpeed)


def populate_source_obstacle_grids_structured(
    particleState,
    config,
    case_config: CaseConfig,
    u_source_grid: torch.Tensor,
    c_source_grid: torch.Tensor,
):
    source_magnitudes: List[float] = []
    for s, source in enumerate(case_config.sources):
        shape_spec = _legacy_to_shape_spec(source, default_kind="sphere")
        sampled = sample_shape_structured(particleState, config, shape_spec)
        u_source_grid = torch.where(sampled != 0, torch.full_like(u_source_grid, s + 1), u_source_grid)

        magnitude = getattr(source, "magnitude", None)
        if magnitude is not None:
            value = float(magnitude)
        elif getattr(source, "randomizeMagnitude", False):
            lo, hi = case_config.amplitudeRange
            value = torch.rand((1,), device=particleState.positions.device).item() * (hi - lo) + lo
        else:
            value = float(case_config.defaultAmplitude)
        source_magnitudes.append(value)

    obstacle_speeds: List[float] = []
    obstacle_counter = 0
    for obstacle in case_config.obstacles:
        shape_spec = _legacy_to_shape_spec(obstacle, default_kind="box")
        kind = normalize_shape_kind(shape_spec.kind)
        params = shape_spec.params or {}

        if kind == "prism" and bool(params.get("add_wall", True)):
            wall_offset = tuple(params.get("wall_offset", shape_spec.position))
            wall_thickness = float(params.get("wall_thickness", 0.04))
            wall = sample_vertical_line(
                particleState,
                config,
                thickness=wall_thickness,
                offset=wall_offset,
            )
            c_source_grid = torch.where(wall != 0, torch.full_like(c_source_grid, -1), c_source_grid)

        sampled = sample_shape_structured(particleState, config, shape_spec)
        as_boundary = bool(params.get("as_boundary", kind in ("single_slit", "double_slit")))

        if as_boundary:
            c_source_grid = torch.where(sampled != 0, torch.full_like(c_source_grid, -1), c_source_grid)
            continue

        obstacle_counter += 1
        c_source_grid = torch.where(sampled != 0, torch.full_like(c_source_grid, obstacle_counter), c_source_grid)

        speed = getattr(obstacle, "speed", None)
        if speed is not None:
            speed_value = float(speed)
        elif getattr(obstacle, "randomizeSpeed", False):
            lo, hi = case_config.obstacleSpeedRange
            speed_value = torch.rand((1,), device=particleState.positions.device).item() * (hi - lo) + lo
        else:
            speed_value = _default_obstacle_speed(case_config, obstacle_counter - 1)
        obstacle_speeds.append(speed_value)

    return u_source_grid, c_source_grid, source_magnitudes, obstacle_speeds


# Backward-compatible names used in existing notebook code.
def sampleShapeStructured(particleState, config, shape: ShapeSpec):
    return sample_shape_structured(particleState, config, shape)


def populateSourceObstacleGridsStructured(particleState, config, caseConfig, uSourceGrid, cSourceGrid):
    return populate_source_obstacle_grids_structured(particleState, config, caseConfig, uSourceGrid, cSourceGrid)
