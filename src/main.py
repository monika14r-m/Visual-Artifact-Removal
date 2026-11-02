"""Command-line interface for classical and ONNX image restoration."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.core.image_loader import load_image, save_image
from src.core.remover import restore_image
from src.core.utils import list_images
from src.metrics import compare_quality


def build_parser():
    parser = argparse.ArgumentParser(
        description="Restore images with OpenCV or an optional ONNX model."
    )
    parser.add_argument("input", nargs="?", help="Input image path")
    parser.add_argument("output", nargs="?", help="Output image path")
    parser.add_argument(
        "--operation",
        choices=["denoise", "deblock", "enhance", "inpaint", "ai-upscale", "ai-inpaint"],
        default="denoise",
    )
    parser.add_argument("--mask", help="White-on-black mask path (required for inpainting)")
    parser.add_argument("--model", help="ONNX model path (required for AI operations)")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="ONNX Runtime execution provider; CUDA requires onnxruntime-gpu",
    )
    parser.add_argument("--method", choices=["telea", "ns"], default="telea")
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--strength", type=float, default=7.0)
    parser.add_argument("--reference", help="Optional clean reference image for PSNR/SSIM")
    parser.add_argument(
        "--batch", nargs=2, metavar=("INPUT_DIR", "OUTPUT_DIR"),
        help="Process supported images in a directory",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def process_one(input_path, output_path, args):
    image = load_image(input_path)
    mask = None
    if args.operation in {"inpaint", "ai-inpaint"}:
        if not args.mask:
            raise ValueError("--mask is required for inpainting")
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {args.mask}")

    if args.operation in {"ai-upscale", "ai-inpaint"}:
        if not args.model:
            raise ValueError("--model is required for AI operations")
        from src.ai.onnx_restorer import ONNXRestorer

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if args.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        task = "upscale" if args.operation == "ai-upscale" else "inpaint"
        engine = ONNXRestorer(args.model, task=task, providers=providers)
        result = engine.upscale(image) if task == "upscale" else engine.inpaint(image, mask)
    else:
        result = restore_image(
            image,
            operation=args.operation,
            mask=mask,
            strength=args.strength,
            method=args.method,
            radius=args.radius,
        )

    saved = save_image(result, output_path, jpeg_quality=args.jpeg_quality)
    report = {
        "input": str(input_path),
        "output": str(saved),
        "operation": args.operation,
        "output_shape": list(result.shape),
    }
    if args.reference:
        reference = load_image(args.reference)
        report["quality_vs_reference"] = compare_quality(reference, result)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch:
        input_dir, output_dir = map(Path, args.batch)
        paths = list_images(input_dir, recursive=args.recursive)
        if not paths:
            raise SystemExit(f"No supported images found in {input_dir}")
        output_root = output_dir.expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        reports = []
        for path in paths:
            source = Path(path)
            target = output_root / f"{source.stem}-restored.png"
            try:
                reports.append(process_one(source, target, args))
                print(f"[OK] {source.name} -> {target}")
            except Exception as exc:
                reports.append({"input": str(source), "error": str(exc)})
                print(f"[ERROR] {source.name}: {exc}")
        success_count = sum("error" not in item for item in reports)
        print(f"Processed {success_count}/{len(reports)} images.")
        return reports

    if not args.input or not args.output:
        parser.error("input and output are required unless --batch is used")
    report = process_one(args.input, args.output, args)
    print(f"[OK] Saved {report['output']}")
    if "quality_vs_reference" in report:
        metrics = report["quality_vs_reference"]
        print(f"PSNR: {metrics['psnr_db']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
    return report


if __name__ == "__main__":
    main()
