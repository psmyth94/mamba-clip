# type: ignore
import argparse

from mamba_clip.pipeline import pipeline, ray_tune_pipeline
from mamba_clip.utils.logging import logger_setup


def arg_parser():
    parser = argparse.ArgumentParser(description="Argument parser for training script")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/isic-2024-challenge/",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Whether to train the model in zero-shot mode",
    )
    parser.add_argument(
        "--val-data-path", type=str, help="Path to the validation dataset"
    )
    parser.add_argument(
        "--train-num-samples", type=int, help="Number of training samples"
    )
    parser.add_argument(
        "--val-num-samples", type=int, help="Number of validation samples"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes for classification (Stage 2 only)",
    )
    parser.add_argument("--sampling", type=str, help="Sampling method")
    parser.add_argument(
        "--undersample",
        type=int,
        help="Number of samples to undersample the majority class to",
    )
    parser.add_argument(
        "--undersample-by",
        type=str,
        help="If included, will take the first --undersample samples from each class."
        " To specify a sort, use --undersample-sort-by in conjunction with this flag.",
    )
    parser.add_argument(
        "--undersample-sort-by",
        type=str,
        help="Column to sort by when undersampling (asc or desc)",
    )

    parser.add_argument(
        "--add-remaining-samples",
        action="store_true",
        help="Add remaining samples removed from undersampling to the validation set",
    )
    parser.add_argument("--balanced-mixup", type=float, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use for training"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs/", help="Directory to save logs"
    )
    parser.add_argument("--log-local", action="store_true", help="Log local ranks")
    parser.add_argument("--name", type=str, help="Name of the experiment")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument("--epochs-cooldown", type=int, help="Number of cooldown epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="Epsilon for Adam optimizer"
    )
    parser.add_argument("--wd", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--warmup",
        type=float,
        default=1,
        help="Warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--use-bn-sync",
        action="store_true",
        help="Use synchronized batch normalization",
    )
    parser.add_argument(
        "--skip-scheduler", action="store_true", help="Skip learning rate scheduler"
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr-restart-interval",
        type=int,
        help="Number of steps before restarting the learning rate scheduler",
    )
    parser.add_argument(
        "--lr-cooldown-end",
        type=float,
        default=0.0,
        help="Learning rate at the end of cooldown",
    )
    parser.add_argument(
        "--lr-cooldown-power",
        type=float,
        default=1.0,
        help="Power for learning rate cooldown",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="Frequency of saving checkpoints"
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        help="Save the most recent checkpoint",
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="Frequency of validation"
    )
    parser.add_argument("--resume", type=str, help="Path to resume checkpoint")
    parser.add_argument(
        "--precision", type=str, default="amp", help="Precision for training"
    )
    parser.add_argument("--stage", type=int, default=1, help="Training stage")
    parser.add_argument(
        "--model-stage-1",
        type=str,
        default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        help="Model for stage 1",
    )
    parser.add_argument(
        "--model-stage-2", type=str, default="ClipClassifier", help="Model for stage 2"
    )
    parser.add_argument(
        "--use-inner-prod",
        action="store_true",
        help=(
            "Whether to use inner product for stage 2 "
            "or the default linear layer for classification"
        ),
    )
    parser.add_argument(
        "--use-visual-only",
        action="store_true",
        help=(
            "Whether to use only the visual features for stage 2 "
            "or both visual and text features"
        ),
    )
    parser.add_argument(
        "--use-text-only",
        action="store_true",
        help=(
            "Whether to use only the text features for stage 2 "
            "or both visual and text features"
        ),
    )
    parser.add_argument(
        "--use-original-model",
        action="store_true",
        help=(
            "Whether to use the fine-tuned model "
            "or the original pretrained model for stage 2",
        ),
    )
    parser.add_argument("--tokenizer", type=str, help="Tokenizer to use")
    parser.add_argument(
        "--lock-image", action="store_true", help="Lock the image tower of the model"
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Number of unlocked groups in the image tower",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        action="store_true",
        help="Freeze batch normalization statistics in the image tower",
    )
    parser.add_argument(
        "--image-mean",
        type=float,
        nargs="+",
        help="Mean values for image normalization",
    )
    parser.add_argument(
        "--image-std",
        type=float,
        nargs="+",
        help="Standard deviation values for image normalization",
    )
    parser.add_argument(
        "--image-interpolation",
        type=str,
        help="Interpolation method for image resizing",
    )
    parser.add_argument("--image-resize-mode", type=str, help="Resize mode for images")
    parser.add_argument(
        "--aug-cfg", type=str, nargs="+", help="Augmentation configuration"
    )
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--local-loss",
        action="store_true",
        help="Use local loss for distributed training",
    )
    parser.add_argument(
        "--gather-with-grad", action="store_true", help="Gather gradients with features"
    )
    parser.add_argument(
        "--force-image-size", type=int, nargs="+", help="Force image size"
    )
    parser.add_argument(
        "--force-quick-gelu", action="store_true", help="Force Quick GELU activation"
    )
    parser.add_argument(
        "--force-patch-dropout", type=float, help="Force patch dropout rate"
    )
    parser.add_argument(
        "--force-custom-text", action="store_true", help="Force custom text model"
    )
    parser.add_argument("--torchscript", action="store_true", help="Use TorchScript")
    parser.add_argument("--torchcompile", action="store_true", help="Use Torch compile")
    parser.add_argument("--trace", action="store_true", help="Enable tracing")
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Frequency of gradient accumulation"
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="env://",
        help="URL for distributed training setup",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        help="Backend for distributed training",
    )
    parser.add_argument("--report-to", type=str, default="", help="Reporting tool")
    parser.add_argument("--wandb-notes", type=str, help="WandB notes")
    parser.add_argument(
        "--wandb-project-name", type=str, default="open-clip", help="WandB project name"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode")
    parser.add_argument(
        "--copy-codebase", action="store_true", help="Copy codebase for reproducibility"
    )
    parser.add_argument(
        "--ddp-static-graph", action="store_true", help="Enable DDP static graph"
    )
    parser.add_argument(
        "--no-set-device-rank", action="store_true", help="Do not set device rank"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--grad-clip-norm", type=float, help="Gradient clipping norm")
    parser.add_argument(
        "--lock-text", action="store_true", help="Lock the text tower of the model"
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Number of unlocked layers in the text tower",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        action="store_true",
        help="Freeze layer normalization in the text tower",
    )
    parser.add_argument(
        "--log-every-n-steps", type=int, default=100, help="Logging frequency"
    )
    parser.add_argument(
        "--hyperparameter-tuning",
        action="store_true",
        help="Enable hyperparameter tuning",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=30,
        help="Number of training iterations for hyperparameter tuning",
    )
    parser.add_argument(
        "--eval-loss",
        type=str,
        help="Evaluation loss to optimize for hyperparameter tuning",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="mamba_clip_study",
        help="Name of the hyperparameter tuning study",
    )
    parser.add_argument(
        "--class-weighted-loss",
        action="store_true",
        help="Use class loss weight for stage 2",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight for COCA caption loss",
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight for COCA contrastive loss",
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument("--remote-sync", type=str, help="Remote sync path")
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="Remote sync frequency in seconds",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        type=str,
        default="fsspec",
        help="Remote sync protocol",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        action="store_true",
        help="Delete previous checkpoints",
    )
    parser.add_argument("--use-bnb-linear", type=str, help="Use BnB linear layer")
    parser.add_argument("--siglip", action="store_true", help="Use SigLIP model")
    parser.add_argument(
        "--small-test", action="store_true", help="Use small dataset for debugging"
    )

    args = parser.parse_args()
    return args


def main():
    logger_setup()
    args = arg_parser()
    args.stage = 1
    args.model_stage_1 = "medmamba"
    args.small_test = True
    if args.hyperparameter_tuning:
        ray_tune_pipeline(args)
    else:
        pipeline(args)


if __name__ == "__main__":
    main()
