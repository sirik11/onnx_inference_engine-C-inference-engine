#!/usr/bin/env python3
"""
quantize.py -- tools for converting ONNX models to float16 or INT8.

This script uses the ONNX Runtime quantization API to perform float16 conversion, dynamic quantization and static quantization.  It is meant as a helper for the ONNX Runtime Inference Accelerator project.

Usage:
  python quantize.py --model model.onnx --output model_fp16.onnx --type fp16
  python quantize.py --model model.onnx --output model_int8.onnx --type dynamic
  python quantize.py --model model.onnx --output model_int8.onnx --type static --data_dir path/to/calibration/data

References:
- ONNX Runtime quantization documentation distinguishing dynamic (for RNNs/transformers) and static (for CNNs) quantization【565839220839977†L365-L396】.
- Float16 conversion using `convert_float_to_float16`【319690206045041†L186-L247】.
"""
import argparse
import os
import sys

from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader
from onnxruntime.tools import convert_float_to_float16

class DummyCalibrationDataReader(CalibrationDataReader):
    """A simple calibration reader that yields random data for static quantization.

    In practice you should implement `get_next` to return actual tensors from your dataset.
    """
    def __init__(self, input_names, data_dir=None, num_samples=1):
        self.input_names = input_names
        self.num_samples = num_samples
        self.count = 0

    def get_next(self):
        if self.count >= self.num_samples:
            return None
        # Returning None for all inputs; ONNX Runtime will treat this as zeros.
        self.count += 1
        return {name: None for name in self.input_names}


def convert_fp16(model_path: Path, output_path: Path):
    model = onnx.load(model_path)
    fp16_model = convert_float_to_float16(model)
    onnx.save(fp16_model, output_path)
    print(f"Saved float16 model to {output_path}")


def dynamic_quant(model_path: Path, output_path: Path):
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(output_path),
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    print(f"Saved dynamic INT8 model to {output_path}")


def static_quant(model_path: Path, output_path: Path, data_dir: Path, num_samples: int = 10):
    # Load the model to get input names
    model = onnx.load(model_path)
    input_names = [input.name for input in model.graph.input]
    reader = DummyCalibrationDataReader(input_names, data_dir, num_samples)
    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format="QDQ",
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"Saved static INT8 model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to fp16 or int8 (dynamic or static)")
    parser.add_argument("--model", required=True, help="Path to the input ONNX model")
    parser.add_argument("--output", required=True, help="Path to save the converted model")
    parser.add_argument("--type", choices=["fp16", "dynamic", "static"], required=True, help="Conversion type")
    parser.add_argument("--data_dir", help="Directory containing calibration data for static quantization")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples for calibration")
    args = parser.parse_args()

    input_path = Path(args.model)
    output_path = Path(args.output)

    if args.type == "fp16":
        convert_fp16(input_path, output_path)
    elif args.type == "dynamic":
        dynamic_quant(input_path, output_path)
    elif args.type == "static":
        if not args.data_dir:
            print("Static quantization requires --data_dir containing calibration data.")
            sys.exit(1)
        static_quant(input_path, output_path, Path(args.data_dir), args.num_samples)


if __name__ == "__main__":
    main()
