import sys
def main():
    try:
        import torch
        print("torch_version=", torch.__version__)
        print("cuda_available=", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda_device_count=", torch.cuda.device_count())
            print("cuda_device_name=", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch_check_error=", repr(e))
        sys.exit(1)
if __name__ == "__main__":
    main()

