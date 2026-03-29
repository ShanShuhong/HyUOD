import argparse
import sys
import os

def run_prep(args):
    print("Running Data Preparation (Generating t and a)...")
    from ta_generate import main as prep_main
    # Mocking sys.argv for the submodule
    sys.argv = ['ta_generate.py', args.images_path, args.output_root]
    prep_main()

def run_train(args):
    print("Running Training...")
    # Using os.system to avoid complex sys.argv manipulation and environment issues
    cmd = f"python train.py {args.model_yaml} {args.data_yaml} --epochs {args.epochs} --batch {args.batch} --imgsz {args.imgsz} --device {args.device} --optimizer {args.optimizer}"
    if args.resume:
        cmd += " --resume"
    if args.pretrained:
        cmd += " --pretrained"
    os.system(cmd)

def run_val(args):
    print("Running Validation...")
    cmd = f"python val.py {args.weights} {args.data_yaml} --imgsz {args.imgsz} --device {args.device} --split {args.split}"
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description="HY-UOD Unified Entry Point")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prep command
    prep_parser = subparsers.add_parser("prep", help="Data preparation (ta_generate)")
    prep_parser.add_argument("images_path", type=str, help="Path to images directory")
    prep_parser.add_argument("output_root", type=str, help="Path to dataset root directory")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("model_yaml", type=str, help="Model YAML")
    train_parser.add_argument("data_yaml", type=str, help="Dataset YAML")
    train_parser.add_argument("--epochs", type=int, default=400)
    train_parser.add_argument("--batch", type=int, default=32)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--device", type=str, default="0")
    train_parser.add_argument("--optimizer", type=str, default="SGD")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--pretrained", action="store_true")

    # Val command
    val_parser = subparsers.add_parser("val", help="Evaluate the model")
    val_parser.add_argument("weights", type=str, help="Weights file path")
    val_parser.add_argument("data_yaml", type=str, help="Dataset YAML")
    val_parser.add_argument("--imgsz", type=int, default=640)
    val_parser.add_argument("--device", type=str, default="0")
    val_parser.add_argument("--split", type=str, default="val")

    args = parser.parse_args()

    if args.command == "prep":
        run_prep(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "val":
        run_val(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
