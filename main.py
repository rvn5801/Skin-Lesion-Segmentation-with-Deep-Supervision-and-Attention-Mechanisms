import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="ISIC 2018 Skin Lesion Segmentation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    train_parser.add_argument("--img_size", type=int, default=None, help="Image size")
    # train_parser.add_argument("--loss", type=str, default="bce_dice", 
    #                     choices=["dice", "bce_dice", "tversky", "focal_tversky"],
    #                     help="Loss function to use")
    train_parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    train_parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable data augmentation")
    train_parser.add_argument("--debug", action="store_true", help="Enable debug mode (small dataset)")
    train_parser.add_argument("--loss", type=str, default="bce_dice", 
                        choices=["dice", "bce_dice", "tversky", "focal_tversky", "combined"],
                        help="Loss function to use")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction on images")
    predict_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    predict_parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    predict_parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
    predict_parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation")
    predict_parser.add_argument("--img_size", type=int, default=256, help="Image size for model input")
    predict_parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference on a single image")
    inference_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    inference_parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    inference_parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save results")
    inference_parser.add_argument("--img_size", type=int, default=256, help="Image size for model input")
    inference_parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation")
    inference_parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.command == "train":
        # Import train module and run
        from train import parse_args as train_parse_args, setup_config, train
        train_args = train_parse_args()
        # Override train_args with main args
        for arg in vars(args):
            if arg != "command" and getattr(args, arg) is not None:
                setattr(train_args, arg, getattr(args, arg))
        
        config = setup_config(train_args)
        train(config, loss_type=train_args.loss, debug=train_args.debug)
        
    elif args.command == "predict":
        # Import predict module and run
        from predict import main as predict_main
        sys.argv = [sys.argv[0]] + [f"--{k}={v}" if not isinstance(v, bool) else f"--{k}" for k, v in vars(args).items() 
                                   if k != "command" and v is not None]
        predict_main()
        
    elif args.command == "inference":
        # Import inference module and run
        from inference import main as inference_main
        sys.argv = [sys.argv[0]] + [f"--{k}={v}" if not isinstance(v, bool) else f"--{k}" for k, v in vars(args).items() 
                                   if k != "command" and v is not None]
        inference_main()
    else:
        print("Please specify a command: train, predict, or inference")

if __name__ == "__main__":
    main()