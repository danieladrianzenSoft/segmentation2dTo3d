import os
import shutil
import json

def flatten_directory(directory, output_directory, file_types, excluded_directories, excluded_files):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    total_files = 0
    copied_files = 0
    skipped_files = []

    def should_exclude(path, filename):
        path_lower = path.lower()
        filename_lower = filename.lower()

        for excluded_dir in excluded_directories:
            if excluded_dir.lower() in path_lower:
                return f"{filename} in excluded directory {excluded_dir}"

        for excluded_str in excluded_files:
            if excluded_str.lower() in filename_lower:
                return f"{filename} contains excluded string {excluded_str}"

        return None

    seen_filenames = set()

    for root, dirs, files in os.walk(directory):
        total_files += len(files)
        dirs[:] = [d for d in dirs if not any(excluded.lower() in d.lower() for excluded in excluded_directories)]

        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()

            if file_extension not in file_types:
                skipped_files.append({"file": file_path, "reason": f"{file_extension} not valid file type"})
                continue

            exclusion_reason = should_exclude(root, file)
            if exclusion_reason:
                skipped_files.append({"file": file_path, "reason": exclusion_reason})
                continue

            if file in seen_filenames:
                skipped_files.append({"file": file_path, "reason": f"{file} already copied"})
                continue

            try:
                shutil.copy(file_path, os.path.join(output_directory, file))
                copied_files += 1
                seen_filenames.add(file)
            except Exception as e:
                skipped_files.append({"file": file_path, "reason": f"unknown error: {str(e)}"})

    summary = {
        "total_files": total_files,
        "copied_files": copied_files,
        "skipped_files": len(skipped_files),
        "details": skipped_files
    }

    summary_file = os.path.join(output_directory, "flatten_summary.json")
    with open(summary_file, "w") as summary_fp:
        json.dump(summary, summary_fp, indent=4)

    print(f"Flattening completed. Summary written to {summary_file}")