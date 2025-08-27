import os
import srt
import argparse
import re

# Define the main directory for output subtitles
OUTPUT_DIR = 'output_subtitles'

def merge_subtitles(directory_name, lang_code):
    """
    Merges all chunk_*.srt files in the specified folder.
    Optionally extracts only a specific language from bilingual subtitles.

    Args:
        directory_name (str): The name of the subdirectory under OUTPUT_DIR.
        lang_code (str): 'all', 'en', or 'zh' to specify which language to keep.
    """
    target_dir = os.path.join(OUTPUT_DIR, directory_name)

    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    chunk_files = [f for f in os.listdir(target_dir) if re.match(r'chunk_(\d+)\.srt', f)]

    if not chunk_files:
        print(f"No 'chunk_*.srt' files found in directory '{target_dir}'.")
        return

    # Sort files intelligently based on the number in the filename
    chunk_files.sort(key=lambda f: int(re.search(r'chunk_(\d+)\.srt', f).group(1)))

    print(f"Found {len(chunk_files)} subtitle chunks. Merging in the following order:")
    for fname in chunk_files:
        print(f" - {fname}")

    all_subtitles = []
    for filename in chunk_files:
        try:
            with open(os.path.join(target_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                subtitles = list(srt.parse(content))
                if subtitles:
                    all_subtitles.extend(subtitles)
        except Exception as e:
            print(f"Error reading or parsing file '{filename}': {e}")
            continue
    
    if not all_subtitles:
        print("Could not parse any subtitle content from the files.")
        return

    # Process subtitles based on the language code
    for sub in all_subtitles:
        if lang_code != 'all' and '\n' in sub.content:
            parts = sub.content.split('\n', 1)
            if lang_code == 'en':
                sub.content = parts[0]
            elif lang_code == 'zh':
                sub.content = parts[1] if len(parts) > 1 else ''
    
    # Re-index all subtitles
    for i, sub in enumerate(all_subtitles):
        sub.index = i + 1

    # Define the output filename with language code
    if lang_code == 'all':
        output_filename = f"{directory_name}_merged.srt"
    else:
        output_filename = f"{directory_name}_merged_{lang_code}.srt"
        
    output_path = os.path.join(target_dir, output_filename)

    try:
        # Compose the final merged subtitle content
        merged_content = srt.compose(all_subtitles)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        print(f"\nMerge successful! Final file saved to: {output_path}")
    except Exception as e:
        print(f"Error saving the merged file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Manually merge all subtitle chunks (chunk_*.srt) in a specified output folder, with an option to extract a single language."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The name of the subdirectory in 'output_subtitles' containing the subtitle chunks (e.g., 'PVPpractise')"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        choices=['all', 'en', 'zh'],
        help="Language to extract: 'en' for English, 'zh' for Chinese, 'all' for both (default: all)"
    )

    args = parser.parse_args()
    merge_subtitles(args.directory, args.lang)
