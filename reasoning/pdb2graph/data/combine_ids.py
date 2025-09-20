import os
import glob
import csv

def process_files_advanced(input_pattern, output_file, items_per_file=100, 
                          output_format='lines', include_filenames=False):
    """
    Advanced version with more options.
    
    Args:
        input_pattern: Pattern to match input files
        output_file: Path to output file
        items_per_file: Number of items to extract from each file
        output_format: 'lines' (one item per line) or 'csv' (comma-separated)
        include_filenames: Whether to include source filename as header
    """
    
    files = glob.glob(input_pattern)
    
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(files)} files to process")
    
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        if output_format == 'csv':
            writer = csv.writer(outfile)
        
        for file_path in files:
            try:
                print(f"Processing: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read().strip()
                    items = content.split(',')[:items_per_file]
                    
                    # Clean items (remove extra whitespace)
                    items = [item.strip() for item in items if item.strip()]
                    
                    if include_filenames:
                        if output_format == 'lines':
                            outfile.write(f"\n# Source: {file_path}\n")
                        else:
                            outfile.write(f"# Source: {file_path}\n")
                    
                    if output_format == 'lines':
                        for item in items:
                            outfile.write(item + '\n')
                    else:  # csv format
                        # writer.writerow([file_path] + items)
                        writer.writerow(items)
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Processing complete. Results written to {output_file}")

# Example usage
if __name__ == "__main__":
    # Basic usage - one item per line
    # process_files_advanced("*.txt", "output_lines.txt")
    
    # CSV format with filenames
    # process_files_advanced("./reasoning/pdb2graph/data/*.txt", "output_with_filenames.csv", 
    #                       output_format='csv', include_filenames=True)
    process_files_advanced("./reasoning/pdb2graph/data/*.txt", "./reasoning/pdb2graph/data/mixed_ids.csv", 
                          output_format='csv', include_filenames=False)