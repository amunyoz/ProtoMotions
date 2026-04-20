#!/usr/bin/env python3
"""Convert local G1 CSV motions into ProtoMotions .motion files.

This is a minimal compatibility wrapper around the existing ProtoMotions G1 CSV
conversion path. The upstream repo already contains
`data/scripts/convert_g1_csv_to_proto.py`, but the verified `robotlab311`
environment intentionally removes `typer` to avoid an Isaac Sim 5.1 click
conflict. This script keeps the same conversion logic while using `argparse`
only, so custom CSV motions can still be imported into the verified stack.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from protomotions.robot_configs.factory import robot_config


PROTOMOTIONS_REPO = Path(os.environ.get("PROTOMOTIONS_REPO", "/home/amunyoz/ProtoMotions"))


def euler_to_quat_wxyz(euler_deg: np.ndarray, order: str) -> np.ndarray:
    rot = Rotation.from_euler(order, euler_deg, degrees=True)
    xyzw = rot.as_quat()
    return np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=-1)


def process_csv_file(
    csv_path: Path,
    *,
    input_fps: int,
    output_fps: int,
    euler_order: str,
    pos_units: str,
    rot_format: str,
    joint_units: str,
    has_header: bool,
    has_frame_column: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    skip = 1 if has_header else 0
    data = np.loadtxt(csv_path, delimiter=",", skiprows=skip)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    col = 1 if has_frame_column else 0

    root_pos = data[:, col : col + 3]
    if pos_units == "cm":
        root_pos = root_pos / 100.0
    col += 3

    if rot_format == "euler_deg":
        root_rot_wxyz = euler_to_quat_wxyz(data[:, col : col + 3], euler_order)
        col += 3
    elif rot_format == "quat_wxyz":
        root_rot_wxyz = data[:, col : col + 4]
        col += 4
    else:
        raise ValueError(f"Unsupported rot_format: {rot_format}")

    joint_angles = data[:, col:]
    if joint_units == "deg":
        joint_angles = np.deg2rad(joint_angles)

    factor = input_fps // output_fps
    if factor > 1:
        root_pos = root_pos[::factor]
        root_rot_wxyz = root_rot_wxyz[::factor]
        joint_angles = joint_angles[::factor]

    return (
        torch.from_numpy(root_pos).to(device=device, dtype=dtype),
        torch.from_numpy(root_rot_wxyz).to(device=device, dtype=dtype),
        torch.from_numpy(joint_angles).to(device=device, dtype=dtype),
    )


def compute_contact_labels_from_pos_and_vel(
    positions: torch.Tensor,
    velocity: torch.Tensor,
    *,
    vel_thres: float,
    height_thresh: float,
) -> torch.Tensor:
    body_height = positions[..., 2]
    body_speed = torch.linalg.norm(velocity, dim=-1)
    return (body_height <= height_thresh) & (body_speed <= vel_thres)


def collect_csv_files(input_path: Path) -> tuple[Path, list[Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got: {input_path}")
        return input_path.parent, [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    csv_files = sorted(input_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found under: {input_path}")
    return input_path, csv_files


def layout_args(layout: str) -> dict[str, object]:
    if layout == "seed":
        return {
            "pos_units": "cm",
            "rot_format": "euler_deg",
            "joint_units": "deg",
            "has_header": True,
            "has_frame_column": True,
        }
    if layout == "kimodo":
        return {
            "pos_units": "m",
            "rot_format": "quat_wxyz",
            "joint_units": "rad",
            "has_header": False,
            "has_frame_column": False,
        }
    raise ValueError(f"Unsupported layout: {layout}")


def infer_layout(input_path: Path) -> str:
    with input_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if any(ch.isalpha() for ch in first_line):
        return "seed"
    return "kimodo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local G1 CSV files into ProtoMotions .motion files."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to one CSV file or a directory tree of CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where converted .motion files will be written.",
    )
    parser.add_argument(
        "--csv-layout",
        choices=("auto", "seed", "kimodo"),
        default="auto",
        help="CSV layout. 'seed' is BONES-SEED style, 'kimodo' is headerless/root-quat format.",
    )
    parser.add_argument("--input-fps", type=int, default=30)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--robot-type", default="g1")
    parser.add_argument("--euler-order", default="xyz")
    parser.add_argument("--ignore-first-n-frames", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing converted .motion files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.chdir(PROTOMOTIONS_REPO)

    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_fps % args.output_fps != 0:
        raise ValueError(
            f"input_fps ({args.input_fps}) must be divisible by output_fps ({args.output_fps})"
        )

    device = torch.device("cpu")
    dtype = torch.float32

    root_input, csv_files = collect_csv_files(input_path)

    robot_mjcf_mapping = {
        "g1": "g1_bm_box_feet.xml",
        "h1_2": "h1_2.xml",
    }
    mjcf_filename = robot_mjcf_mapping.get(args.robot_type, f"{args.robot_type}.xml")
    mjcf_path = PROTOMOTIONS_REPO / "protomotions" / "data" / "assets" / "mjcf" / mjcf_filename
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF file not found at {mjcf_path}")

    kinematic_info = extract_kinematic_info(str(mjcf_path))
    robot_cfg = robot_config(args.robot_type)
    left_foot_name = robot_cfg.common_naming_to_robot_body_names["all_left_foot_bodies"][0]
    right_foot_name = robot_cfg.common_naming_to_robot_body_names["all_right_foot_bodies"][0]
    body_names = kinematic_info.body_names
    left_foot_idx = body_names.index(left_foot_name)
    right_foot_idx = body_names.index(right_foot_name)

    print(f"Robot type: {args.robot_type}")
    print(f"Input root: {root_input}")
    print(f"Output dir: {output_dir}")
    print(f"Left foot: {left_foot_name} (index {left_foot_idx})")
    print(f"Right foot: {right_foot_name} (index {right_foot_idx})")

    with torch.no_grad():
        for csv_path in csv_files:
            relative_path = csv_path.relative_to(root_input) if csv_path != root_input else Path(csv_path.name)
            outpath = (output_dir / relative_path).with_suffix(".motion")
            outpath.parent.mkdir(parents=True, exist_ok=True)

            if outpath.exists() and not args.force:
                print(f"Skipping existing motion: {outpath}")
                continue

            csv_layout = args.csv_layout
            if csv_layout == "auto":
                csv_layout = infer_layout(csv_path)
            cfg = layout_args(csv_layout)

            print(f"Converting {csv_path} using layout={csv_layout}")

            root_pos, root_rot_wxyz, joint_angles = process_csv_file(
                csv_path,
                input_fps=args.input_fps,
                output_fps=args.output_fps,
                euler_order=args.euler_order,
                pos_units=cfg["pos_units"],
                rot_format=cfg["rot_format"],
                joint_units=cfg["joint_units"],
                has_header=cfg["has_header"],
                has_frame_column=cfg["has_frame_column"],
                device=device,
                dtype=dtype,
            )

            expected_dofs = kinematic_info.num_dofs
            actual_dofs = joint_angles.shape[-1]
            if actual_dofs != expected_dofs:
                raise ValueError(
                    f"{csv_path.name}: joint angle columns ({actual_dofs}) != expected DOFs ({expected_dofs})"
                )

            if args.ignore_first_n_frames > 0:
                root_pos = root_pos[args.ignore_first_n_frames :]
                root_rot_wxyz = root_rot_wxyz[args.ignore_first_n_frames :]
                joint_angles = joint_angles[args.ignore_first_n_frames :]

            qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)
            root_pos_from_qpos, joint_rot_mats = extract_transforms_from_qpos(kinematic_info, qpos)

            motion = fk_from_transforms_with_velocities(
                kinematic_info=kinematic_info,
                root_pos=root_pos_from_qpos,
                joint_rot_mats=joint_rot_mats,
                fps=args.output_fps,
                compute_velocities=True,
                velocity_max_horizon=3,
            )

            qpos = extract_qpos_from_transforms(kinematic_info, root_pos, joint_rot_mats)
            motion.dof_pos = qpos[:, 7:]
            motion.dof_vel = compute_cartesian_velocity(
                batched_robot_pos=joint_angles.unsqueeze(1),
                fps=args.output_fps,
            ).squeeze(1)

            translation_vecs = motion.fix_height_per_frame(height_offset=0.02)
            if motion.rigid_body_vel is not None and motion.fps is not None:
                vel_delta = torch.zeros(
                    translation_vecs.shape[0],
                    1,
                    3,
                    device=motion.rigid_body_vel.device,
                    dtype=motion.rigid_body_vel.dtype,
                )
                vel_delta[:-1] = (
                    (translation_vecs[1:] - translation_vecs[:-1]).unsqueeze(1) / motion.motion_dt
                )
                motion.rigid_body_vel = motion.rigid_body_vel + vel_delta

            motion.fix_height(height_offset=0.04)
            motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
                positions=motion.rigid_body_pos,
                velocity=motion.rigid_body_vel,
                vel_thres=0.15,
                height_thresh=0.1,
            ).to(torch.bool)
            motion.local_rigid_body_rot = None

            torch.save(motion.to_dict(), str(outpath))
            print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
