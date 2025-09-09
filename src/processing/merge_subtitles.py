import os
import srt
import argparse
import re

# Define the main directory for output subtitles - use dynamic paths
try:
    from src.utils.constants import get_output_dir, OUTPUT_DIR_NAME
    OUTPUT_DIR = OUTPUT_DIR_NAME  # For backward compatibility
except ImportError:
    OUTPUT_DIR = 'output_subtitles'

def merge_subtitles(directory_name, lang_code, subfolder=None):
    """
    Merges all chunk_*.srt files in the specified folder.
    Optionally extracts only a specific language from bilingual subtitles.

    Args:
        directory_name (str): The name of the subdirectory under OUTPUT_DIR.
        lang_code (str): 'all', 'en', or 'zh' to specify which language to keep.
        subfolder (str): Optional subfolder (e.g., 'local_llm', 'api_llm') containing translated chunks.
    """
    target_dir = os.path.join(OUTPUT_DIR, directory_name)
    if subfolder:
        target_dir = os.path.join(target_dir, subfolder)

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
    
    # Re-index all subtitles with timing validation
    prev_end_time = 0.0
    max_timestamp = 0.0
    timing_issues = 0
    
    for i, sub in enumerate(all_subtitles):
        sub.index = i + 1
        
        start_sec = sub.start.total_seconds()
        end_sec = sub.end.total_seconds()
        
        # Check for timing issues
        if start_sec < prev_end_time:
            timing_issues += 1
            if timing_issues <= 3:  # Only log first few issues
                print(f"  ⚠️ Warning: Subtitle {i+1} overlaps with previous: {start_sec:.2f}s < {prev_end_time:.2f}s")
        
        max_timestamp = max(max_timestamp, end_sec)
        prev_end_time = end_sec
    
    if timing_issues > 0:
        print(f"  ⚠️ Total timing issues found: {timing_issues}")
    
    # Log final timing statistics
    total_duration_hours = max_timestamp / 3600.0
    print(f"\n📊 MERGE TIMING ANALYSIS:")
    print(f"  Total subtitles merged: {len(all_subtitles)}")
    print(f"  Maximum timestamp: {max_timestamp:.2f}s ({max_timestamp/60:.1f} minutes, {total_duration_hours:.2f} hours)")
    print(f"  Timing overlaps detected: {timing_issues}")

    # Define the output filename with language code and subfolder
    if subfolder:
        base_name = f"{directory_name}_{subfolder}"
    else:
        base_name = directory_name
        
    if lang_code == 'all':
        output_filename = f"{base_name}_merged.srt"
    else:
        output_filename = f"{base_name}_merged_{lang_code}.srt"
        
    output_path = os.path.join(target_dir, output_filename)

    try:
        # Compose the final merged subtitle content
        merged_content = srt.compose(all_subtitles)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        print(f"\n✅ Merge successful! Final file saved to: {output_path}")
        
        # Final file size check
        try:
            file_size = os.path.getsize(output_path)
            print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        except Exception:
            pass
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
    parser.add_argument(
        "--subfolder",
        type=str,
        help="Subfolder containing translated chunks (e.g., 'local_llm', 'api_llm')"
    )

    args = parser.parse_args()
    merge_subtitles(args.directory, args.lang, args.subfolder)
