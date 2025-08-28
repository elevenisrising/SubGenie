import nltk
import ssl

def download_nltk_data():
    """
    Downloads all NLTK datasets.
    This function includes a workaround for SSL certificate issues.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # This workaround is for older Python versions that don't have this attribute.
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    print("Downloading all NLTK datasets (this may take some time)...")
    try:
        if nltk.download('all'):
            print("\nDownload complete.")
            print("You can now run the main application.")
    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        print("Please check your internet connection and permissions.")
        print("If the download fails, you may need to configure a proxy or check your firewall settings.")

if __name__ == "__main__":
    download_nltk_data()
