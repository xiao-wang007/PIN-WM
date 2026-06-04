"""
Standalone script to train static 2D Gaussian Splatting on a dataset.
Usage:
    python scripts/train_static_2dgs.py \
        --source_path dataset/tomato_can_static \
        --output_path output/tomato_can_static \
        --iterations 5000
"""

import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.getcwd())

from diff_rendering.gaussian_splatting_2d.arguments import ModelParams, PipelineParams, OptimizationParams
from diff_rendering.gaussian_splatting_2d.train_2dgs import train_static_2dgs
from diff_rendering.gaussian_splatting_2d.scene.dataset_readers import sceneLoadTypeCallbacks
from utils.cfg_utils import Gaussian_args


def main():
    parser = ArgumentParser(description="Train static 2D Gaussian Splatting")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to dataset (contains transforms_train.json)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save outputs")
    parser.add_argument("--white_background", action="store_true", default=True,
                        help="Use white background (default: True)")
    parser.add_argument("--iterations", type=int, default=5000,
                        help="Number of training iterations")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="Spherical harmonics degree")
    args = parser.parse_args()

    # Set up the argument groups
    arg_parser = ArgumentParser()
    lp = ModelParams(arg_parser)
    op = OptimizationParams(arg_parser)
    pp = PipelineParams(arg_parser)
    # Parse empty args to get defaults, then override with our values
    parsed = arg_parser.parse_args([])

    parsed.source_path = args.source_path
    parsed.model_path = os.path.join(args.output_path, "static")
    parsed.white_background = args.white_background
    parsed.sh_degree = args.sh_degree

    dataset = lp.extract(parsed)
    opt = op.extract(parsed)
    pipe = pp.extract(parsed)

    # Override iterations
    opt.iterations = args.iterations

    # Load scene info using the Blender/NerfSynthetic reader
    print(f"Loading dataset from: {parsed.source_path}")
    scene_info = sceneLoadTypeCallbacks["Blender"](
        parsed.source_path, parsed.white_background, eval=False
    )

    testing_iterations = [args.iterations, args.iterations // 2]
    saving_iterations = [args.iterations, args.iterations // 2]

    gaussian_args = Gaussian_args(
        dataset, opt, pipe,
        testing_iterations, saving_iterations,
        checkpoint_iterations=[], checkpoint=None
    )

    print(f"Output will be saved to: {parsed.model_path}")
    os.makedirs(parsed.model_path, exist_ok=True)

    gaussians = train_static_2dgs(gaussian_args, scene_info, mesh=None, logger=None)
    print("Training complete!")


if __name__ == "__main__":
    main()
