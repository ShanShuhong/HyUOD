from ultralytics import YOLO
import os
import argparse
if __name__ == "__main__":
    # Initialize the command-line argument parser
    parser = argparse.ArgumentParser(description="YOLO training script with Physics-Coupled Frequency Dynamic Adaptation.")
    
    # Required arguments
    parser.add_argument("model_yaml", type=str, help="Path to the model YAML file (e.g., train_yaml/hyuod.yaml)")
    parser.add_argument("data_yaml", type=str, help="Path to the dataset YAML file")
    
    # Optional hyperparameters
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs (default: 400)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device to run on (e.g., 0 or 0,1,2,3 or cpu)")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], help="Optimizer to use (default: SGD)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    parser.add_argument("--workers", type=int, default=16, help="Number of data loading workers (default: 8)")
    parser.add_argument("--project", type=str, default="runs/train", help="Project name for saving results")
    parser.add_argument("--name", type=str, default="hyuod_experiment", help="Experiment name")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights if available")
    
    # Parse the arguments
    args = parser.parse_args()

    print(f"--- HY-UOD Training ---")
    print(f"Model YAML: {args.model_yaml}")
    print(f"Data YAML: {args.data_yaml}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, Imgsz: {args.imgsz}")
    print(f"Device: {args.device}, Optimizer: {args.optimizer}")
    print(f"-----------------------")

    # Build a new model from the provided YAML path
    model = YOLO(args.model_yaml)

    # Train the model using the parsed data path and parameters
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer=args.optimizer,
        resume=args.resume,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        cache='ram',
        iou=0.4,
        save=True 
    )

# # Load a model
# model = YOLO("/opt/data/private/UOD/ultralytics-mm/train_runs/mm_frequency/3_28_mm_v2_duov2/train2/weights/best.pt")  # build a new model from YAML
# metrics = model.val(iou=0.3)
# print(model.info(detailed=True))

# # # Train the model
# results = model.train(resume=True, epochs=400, batch=16, optimizer='SGD', imgsz=640, device=0, name='test_cuda11_8_89_s_duo')
# results = model.train(data='/opt/data/private/UOD/DUO/duo.yaml', epochs=400, batch=16, optimizer='SGD', pretrained=False, imgsz=640, device=0, name='a6000_brk_2_2_duo')