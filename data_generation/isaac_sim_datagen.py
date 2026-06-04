#!/usr/bin/env python3
"""Generate a static multi-camera Isaac Sim dataset for PIN-WM/2DGS.

Run this with Isaac Sim's bundled Python, not the repo conda env, for example:

    /path/to/isaac-sim/python.sh data_generation/isaac_sim_datagen.py \
        --object-mesh envs/asset/cube_t/cube_t.obj \
        --output-dir dataset/isaac_cube_t_static \
        --num-cameras 12 \
        --headless

The script creates an object on a table, places cameras on a circular orbit,
captures RGB and instance masks, and writes:

    dynamic_train0.json
    transforms_train.json
    dynamic_data_train/train0/r_<camera>_0.png
    dynamic_data_train/train0/m_<camera>_0.png
    scene.usd

`m_*.png` is an RGBA image whose alpha channel is the target-object mask.
That matches the loader behavior in utils/data_utils.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a static multi-camera object-on-table scene in Isaac Sim."
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/isaac_static_object",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--object-usd",
        default=None,
        help="Optional USD/USDZ asset to reference as the target object.",
    )
    parser.add_argument(
        "--object-mesh",
        default=None,
        help="Optional OBJ/STL/PLY mesh to convert into a USD Mesh prim.",
    )
    parser.add_argument(
        "--target-label",
        default="target_object",
        help="Semantic class label used to build m_*.png alpha masks.",
    )
    parser.add_argument(
        "--object-position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Initial object translation before optional table auto-placement.",
    )
    parser.add_argument(
        "--object-rotation-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Object XYZ Euler rotation in degrees.",
    )
    parser.add_argument(
        "--object-scale",
        nargs="+",
        type=float,
        default=(1.0,),
        help="Object scale. Pass one value for uniform or three for XYZ.",
    )
    parser.add_argument(
        "--default-object-size",
        type=float,
        default=0.06,
        help="Cube edge length used when no object asset is provided.",
    )
    parser.add_argument(
        "--auto-place-on-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shift the object so its world-space bottom rests on the table.",
    )
    parser.add_argument(
        "--object-z-offset",
        type=float,
        default=0.0,
        help="Extra Z offset after auto-placement on the table.",
    )
    parser.add_argument(
        "--table-size",
        nargs=3,
        type=float,
        default=(0.8, 0.6, 0.04),
        metavar=("X", "Y", "Z"),
        help="Tabletop dimensions in meters.",
    )
    parser.add_argument(
        "--table-top-z",
        type=float,
        default=0.0,
        help="World-space Z height of the tabletop surface.",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=12,
        help="Number of camera views around the object.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.45,
        help="Camera orbit radius in meters.",
    )
    parser.add_argument(
        "--elevation-deg",
        type=float,
        default=35.0,
        help="Camera elevation angle above the table plane.",
    )
    parser.add_argument(
        "--start-azimuth-deg",
        type=float,
        default=-45.0,
        help="Azimuth of camera 0 in degrees.",
    )
    parser.add_argument(
        "--look-at",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.035),
        metavar=("X", "Y", "Z"),
        help="Camera look-at point in world coordinates.",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=(800, 800),
        metavar=("WIDTH", "HEIGHT"),
        help="Camera resolution.",
    )
    parser.add_argument(
        "--focal-length-mm",
        type=float,
        default=24.0,
        help="USD camera focal length in millimeters.",
    )
    parser.add_argument(
        "--horizontal-aperture-mm",
        type=float,
        default=20.0,
        help="USD camera horizontal aperture in millimeters.",
    )
    parser.add_argument(
        "--vertical-aperture-mm",
        type=float,
        default=None,
        help="USD camera vertical aperture. Defaults to aspect-scaled horizontal aperture.",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.01,
        help="Near clipping plane in meters.",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=20.0,
        help="Far clipping plane in meters.",
    )
    parser.add_argument(
        "--samples-per-pixel",
        type=int,
        default=64,
        help="RTX samples per pixel for cleaner captures.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Isaac/RTX warmup frames before capture.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Isaac Sim without the GUI.",
    )
    parser.add_argument(
        "--renderer",
        default="RaytracedLighting",
        help="Isaac Sim renderer name.",
    )
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector {vector}")
    return vector / norm


def as_xyz_scale(scale_values: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in scale_values)
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) == 3:
        return values
    raise ValueError("--object-scale expects either one value or three XYZ values")


def look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    up_guess: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64),
) -> np.ndarray:
    """Return OpenGL/USD camera-to-world matrix.

    The camera axes are X right, Y up, and Z backward. This is the convention
    expected by this repo's JSON loader before its COLMAP-style Y/Z flip.
    """

    back = normalize(eye - target)
    right = np.cross(up_guess, back)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), back)
    right = normalize(right)
    up = normalize(np.cross(back, right))

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = back
    matrix[:3, 3] = eye
    return matrix


def usd_matrix_from_c2w(c2w: np.ndarray) -> Any:
    """Convert column-vector camera-to-world to USD's row-vector matrix layout."""

    from pxr import Gf

    right = c2w[:3, 0]
    up = c2w[:3, 1]
    back = c2w[:3, 2]
    eye = c2w[:3, 3]
    return Gf.Matrix4d(
        right[0],
        right[1],
        right[2],
        0.0,
        up[0],
        up[1],
        up[2],
        0.0,
        back[0],
        back[1],
        back[2],
        0.0,
        eye[0],
        eye[1],
        eye[2],
        1.0,
    )


def camera_intrinsic(
    width: int,
    height: int,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    vertical_aperture_mm: float | None,
) -> list[list[float]]:
    if vertical_aperture_mm is None:
        vertical_aperture_mm = horizontal_aperture_mm * height / width

    fx = width * focal_length_mm / horizontal_aperture_mm
    fy = height * focal_length_mm / vertical_aperture_mm
    cx = width * 0.5
    cy = height * 0.5
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def create_display_color(prim: Any, color: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    if prim and prim.IsValid() and prim.IsA(UsdGeom.Gprim):
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def set_xform(
    prim: Any,
    translation: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> None:
    from pxr import UsdGeom

    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(translation)
    xform_api.SetRotate(rotation_deg, UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xform_api.SetScale(scale)


def add_semantic_label(prim: Any, label: str) -> None:
    """Apply a semantic class label across Isaac Sim versions."""

    if not prim or not prim.IsValid():
        return

    try:
        from isaacsim.core.utils.semantics import add_labels

        add_labels(prim, labels=[label], instance_name="class", overwrite=True)
        return
    except Exception:
        pass

    try:
        from omni.isaac.core.utils.semantics import add_update_semantics

        add_update_semantics(prim, label, semantic_type="class")
        return
    except Exception:
        pass

    try:
        import omni.replicator.core as rep

        with rep.get.prim_at_path(str(prim.GetPath())):
            rep.modify.semantics([("class", label)])
    except Exception as exc:
        print(f"[WARN] Could not add semantic label to {prim.GetPath()}: {exc}")


def add_semantic_label_recursive(root_prim: Any, label: str) -> None:
    from pxr import Usd, UsdGeom

    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Imageable):
            add_semantic_label(prim, label)


def create_table(stage: Any, args: argparse.Namespace) -> Any:
    from pxr import UsdGeom

    table_path = "/World/Table"
    table_size = tuple(float(value) for value in args.table_size)
    table_center_z = args.table_top_z - table_size[2] * 0.5

    table = UsdGeom.Cube.Define(stage, table_path)
    table.CreateSizeAttr(1.0)
    set_xform(
        table.GetPrim(),
        translation=(0.0, 0.0, table_center_z),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=table_size,
    )
    create_display_color(table.GetPrim(), (0.45, 0.36, 0.27))
    return table.GetPrim()


def create_default_object(stage: Any, args: argparse.Namespace) -> Any:
    from pxr import UsdGeom

    object_path = "/World/TargetObject"
    cube = UsdGeom.Cube.Define(stage, object_path)
    cube.CreateSizeAttr(args.default_object_size)
    create_display_color(cube.GetPrim(), (0.05, 0.35, 0.9))
    return cube.GetPrim()


def create_usd_object(stage: Any, asset_path: str) -> Any:
    from pxr import UsdGeom

    object_path = "/World/TargetObject"
    object_xform = UsdGeom.Xform.Define(stage, object_path)
    object_xform.GetPrim().GetReferences().AddReference(str(Path(asset_path).expanduser().resolve()))
    return object_xform.GetPrim()


def create_mesh_object(stage: Any, mesh_path: str) -> Any:
    """Create a USD Mesh prim from OBJ/STL/PLY via trimesh."""

    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "Loading --object-mesh requires trimesh inside Isaac Sim's Python. "
            "Install it with Isaac's python.sh -m pip install trimesh, or pass --object-usd."
        ) from exc

    from pxr import UsdGeom

    loaded = trimesh.load(str(Path(mesh_path).expanduser().resolve()), force="scene")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded

    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Mesh has no vertices/faces: {mesh_path}")

    object_xform = UsdGeom.Xform.Define(stage, "/World/TargetObject")
    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/TargetObject/Mesh")
    mesh_prim.CreatePointsAttr([tuple(map(float, vertex)) for vertex in np.asarray(mesh.vertices)])
    mesh_prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    mesh_prim.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces, dtype=np.int64).reshape(-1).tolist())
    mesh_prim.CreateSubdivisionSchemeAttr("none")
    create_display_color(mesh_prim.GetPrim(), (0.05, 0.35, 0.9))
    return object_xform.GetPrim()


def create_object(stage: Any, args: argparse.Namespace) -> Any:
    if args.object_usd and args.object_mesh:
        raise ValueError("Pass only one of --object-usd or --object-mesh")
    if args.object_usd:
        return create_usd_object(stage, args.object_usd)
    if args.object_mesh:
        return create_mesh_object(stage, args.object_mesh)
    return create_default_object(stage, args)


def compute_world_bbox(stage: Any, prim: Any) -> Any:
    from pxr import Usd, UsdGeom

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    return bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()


def place_object(stage: Any, object_prim: Any, args: argparse.Namespace) -> list[float]:
    translation = tuple(float(value) for value in args.object_position)
    rotation = tuple(float(value) for value in args.object_rotation_deg)
    scale = as_xyz_scale(args.object_scale)

    set_xform(object_prim, translation=translation, rotation_deg=rotation, scale=scale)

    if args.auto_place_on_table:
        bbox = compute_world_bbox(stage, object_prim)
        min_z = float(bbox.GetMin()[2])
        translation = (
            translation[0],
            translation[1],
            translation[2] + args.table_top_z - min_z + args.object_z_offset,
        )
        set_xform(object_prim, translation=translation, rotation_deg=rotation, scale=scale)

    return [translation[0], translation[1], translation[2], 0.0, 0.0, 0.0, 1.0]


def create_lighting(stage: Any) -> None:
    from pxr import UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(350.0)

    key = UsdLux.SphereLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(2500.0)
    key.CreateRadiusAttr(0.5)
    set_xform(
        key.GetPrim(),
        translation=(0.3, -0.4, 0.7),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )


def create_camera(stage: Any, cam_path: str, c2w: np.ndarray, args: argparse.Namespace) -> Any:
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, cam_path)
    camera.CreateFocalLengthAttr(args.focal_length_mm)
    camera.CreateHorizontalApertureAttr(args.horizontal_aperture_mm)
    vertical_aperture = args.vertical_aperture_mm
    if vertical_aperture is None:
        width, height = args.resolution
        vertical_aperture = args.horizontal_aperture_mm * height / width
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateClippingRangeAttr(Gf.Vec2f(args.near, args.far))

    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(usd_matrix_from_c2w(c2w))
    return camera.GetPrim()


def create_cameras(stage: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    from pxr import UsdGeom

    UsdGeom.Scope.Define(stage, "/World/Cameras")
    look_at = np.array(args.look_at, dtype=np.float64)
    elevation = math.radians(args.elevation_deg)
    width, height = args.resolution
    intrinsic = camera_intrinsic(
        width=width,
        height=height,
        focal_length_mm=args.focal_length_mm,
        horizontal_aperture_mm=args.horizontal_aperture_mm,
        vertical_aperture_mm=args.vertical_aperture_mm,
    )

    cameras = []
    for cam_id in range(args.num_cameras):
        azimuth = math.radians(args.start_azimuth_deg + cam_id * 360.0 / args.num_cameras)
        eye = look_at + np.array(
            [
                args.radius * math.cos(elevation) * math.cos(azimuth),
                args.radius * math.cos(elevation) * math.sin(azimuth),
                args.radius * math.sin(elevation),
            ],
            dtype=np.float64,
        )
        c2w = look_at_c2w(eye, look_at)
        cam_path = f"/World/Cameras/Camera_{cam_id:02d}"
        create_camera(stage, cam_path, c2w, args)
        cameras.append(
            {
                "id": cam_id,
                "path": cam_path,
                "c2w": c2w,
                "intrinsic": intrinsic,
            }
        )
    return cameras


def configure_render_settings(args: argparse.Namespace) -> None:
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/rtx/pathtracing/spp", args.samples_per_pixel)
    settings.set("/rtx/pathtracing/totalSpp", args.samples_per_pixel)
    settings.set("/rtx/rendermode", "RaytracedLighting")
    # Suppress the counter-performant copy-path performance warning from syntheticdata
    try:
        import omni.log
        omni.log.set_level(omni.log.Level.ERROR, channel="omni.syntheticdata.plugin")
    except Exception:
        pass


def make_instance_annotator(rep: Any) -> Any:
    
    candidates = (
        {"semanticTypes": ["class"], "colorize": False},
        {"semanticTypes": ["class"], "Colorize": False},
        {"colorize": False},
        {"Colorize": False},
        None,
    )
    for init_params in candidates:
        try:
            if init_params is None:
                return rep.annotators.get("instance_segmentation")
            return rep.annotators.get("instance_segmentation", init_params=init_params)
        except Exception:
            continue
    raise RuntimeError("Could not create an Isaac Sim instance_segmentation annotator")


def rgb_to_uint8(data: Any) -> np.ndarray:
    array = np.asarray(data)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    # np.ascontiguousarray ensures a C-contiguous layout; the [..., :3] slice
    # above creates a strided (non-contiguous) view that crashes Isaac's bundled
    # Pillow PNG encoder with "tile cannot extend outside image".
    return np.ascontiguousarray(array)


def extract_target_ids(info: dict[str, Any], object_path: str, target_label: str) -> list[int]:
    target_ids: set[int] = set()

    id_to_labels = info.get("idToLabels", {}) or {}
    for raw_id, labels in id_to_labels.items():
        label_text = json.dumps(labels) if not isinstance(labels, str) else labels
        if object_path in label_text or target_label in label_text:
            target_ids.add(int(raw_id))

    id_to_semantics = info.get("idToSemantics", info.get("idToSemantic", {})) or {}
    for raw_id, semantics in id_to_semantics.items():
        semantic_text = json.dumps(semantics) if not isinstance(semantics, str) else semantics
        if target_label in semantic_text or object_path in semantic_text:
            target_ids.add(int(raw_id))

    return sorted(target_ids)


def mask_from_instance_data(instance_payload: Any, object_path: str, target_label: str) -> np.ndarray:
    if isinstance(instance_payload, dict):
        data = np.asarray(instance_payload.get("data"))
        info = instance_payload.get("info", {}) or {}
    else:
        data = np.asarray(instance_payload)
        info = {}

    if data.ndim == 3:
        if data.shape[-1] == 1:
            data = data[..., 0]
        else:
            raise RuntimeError(
                "Instance segmentation returned a colorized image. "
                "Try a newer Isaac Sim version or lower-level non-colorized annotator settings."
            )

    target_ids = extract_target_ids(info, object_path, target_label)
    if not target_ids:
        print(
            "[WARN] No explicit target-object ID found in instance segmentation; "
            "falling back to all non-background labelled pixels."
        )
        return data > 1

    return np.isin(data, target_ids)


def save_rgba_masked(rgb: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    import cv2
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb[..., :3]
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    # cv2 uses BGRA order; convert RGB→BGR for colour channels
    bgra = rgba[..., [2, 1, 0, 3]]
    cv2.imwrite(str(output_path), bgra)


def capture_dataset(
    simulation_app: Any,
    cameras: list[dict[str, Any]],
    object_pose: list[float],
    args: argparse.Namespace,
) -> None:
    import omni.replicator.core as rep

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "dynamic_data_train" / "train0"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_datas = []
    blender_frames = []
    width, _ = args.resolution
    fx = cameras[0]["intrinsic"][0][0]
    camera_angle_x = 2.0 * math.atan(0.5 * width / fx)

    import cv2

    # Process one camera at a time: create render product → step orchestrator
    # → read data → destroy.  This avoids multi-render-product parallelism
    # issues and ensures each annotator has real rendered data.
    for camera in cameras:
        cam_id = camera["id"]
        rel_rgb_path = f"dynamic_data_train/train0/r_{cam_id}_0.png"
        rgb_path = output_dir / rel_rgb_path
        masked_path = image_dir / f"m_{cam_id}_0.png"

        render_product = rep.create.render_product(
            camera["path"],
            tuple(args.resolution),
            name=f"rp_camera_{cam_id:02d}",
        )
        rgb_annotator = rep.annotators.get("rgb")
        instance_annotator = make_instance_annotator(rep)
        rgb_annotator.attach(render_product)
        instance_annotator.attach(render_product)

        # Warmup for this camera
        for _ in range(max(args.warmup_frames, 2)):
            simulation_app.update()

        # Step the Replicator pipeline — required for annotators to receive data
        rep.orchestrator.step()
        simulation_app.update()

        rgb = rgb_to_uint8(rgb_annotator.get_data())
        mask = mask_from_instance_data(
            instance_annotator.get_data(),
            object_path="/World/TargetObject",
            target_label=args.target_label,
        )

        # cv2 expects BGR; rgb is RGB, so flip channels
        cv2.imwrite(str(rgb_path), rgb[..., ::-1])
        save_rgba_masked(rgb, mask, masked_path)
        print(f"  Captured camera {cam_id}")

        try:
            rgb_annotator.detach([render_product])
            instance_annotator.detach([render_product])
        except Exception:
            pass
        try:
            render_product.destroy()
        except Exception:
            pass

        c2w = camera["c2w"]
        image_datas.append(
            {
                "time": 0.0,
                "c2w": c2w[:3, :].tolist(),
                "intrinsic": camera["intrinsic"],
                "file_path": rel_rgb_path,
            }
        )
        blender_frames.append(
            {
                "file_path": f"dynamic_data_train/train0/m_{cam_id}_0",
                "transform_matrix": c2w.tolist(),
            }
        )

    pinwm_json = {
        "ee_init_position": [0.0, 0.0],
        "ee_goal_position": [0.0, 0.0],
        "image_datas": image_datas,
        "pose_datas": [object_pose],
    }
    with (output_dir / "dynamic_train0.json").open("w", encoding="utf-8") as file:
        json.dump(pinwm_json, file, indent=4)

    blender_json = {
        "camera_angle_x": camera_angle_x,
        "frames": blender_frames,
    }
    with (output_dir / "transforms_train.json").open("w", encoding="utf-8") as file:
        json.dump(blender_json, file, indent=4)
    with (output_dir / "transforms_test.json").open("w", encoding="utf-8") as file:
        json.dump(blender_json, file, indent=4)

    print(f"Wrote {len(cameras)} camera views to {output_dir}")
    print("Use m_*.png for masked 2DGS rendering-alignment supervision.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
        }
    )

    import omni.usd
    from pxr import UsdGeom

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    configure_render_settings(args)
    create_lighting(stage)
    create_table(stage, args)
    object_prim = create_object(stage, args)
    object_pose = place_object(stage, object_prim, args)
    add_semantic_label_recursive(object_prim, args.target_label)
    cameras = create_cameras(stage, args)

    usd_path = output_dir / "scene.usd"
    stage.GetRootLayer().Export(str(usd_path))
    print(f"Saved Isaac scene to {usd_path}")

    capture_dataset(simulation_app, cameras, object_pose, args)
    try:
        import omni.replicator.core as rep
        rep.orchestrator.stop()
    except Exception:
        pass
    simulation_app.close()


if __name__ == "__main__":
    main()
