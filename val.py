import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO validation script for HY-UOD.")
    
    parser.add_argument("weights", type=str, help="Path to the model weights file (e.g., weights/DUO.pt)")
    parser.add_argument("data_yaml", type=str, help="Path to the dataset YAML file")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device to run on (e.g., 0 or cpu)")
    parser.add_argument("--split", type=str, default="val", choices=['val', 'test', 'train'], help="Dataset split to use (default: val)")
    parser.add_argument("--iou", type=float, default=0.4, help="IoU threshold for validation (default: 0.4)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001)")
    
    args = parser.parse_args()
    
    print(f"--- HY-UOD Validation ---")
    print(f"Weights: {args.weights}")
    print(f"Data YAML: {args.data_yaml}")
    print(f"Image Size: {args.imgsz}, Split: {args.split}")
    print(f"--------------------------")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data_yaml, 
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        iou=args.iou,
        conf=args.conf
    )
    print("Validation completed successfully!")

    #python val.py runs/detect/train15/weights/best.pt test_yaml/test.yaml