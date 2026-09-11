"""Compatibility entrypoint for the repaired MVP trainer."""

try:
    from .train_resnet_v2 import main
except ImportError:
    from train_resnet_v2 import main


if __name__ == "__main__":
    main()
