# verify that all files in the specified directories follow the rule of starting with "user:" or "assistant:" outside of <question> tags

import os

disciplines = ["Biology", "Chemistry", "Economics", "Mathematics", "Statistics", "undisciplined"]

for discipline in disciplines:
    # Directory containing the files to check
    directory_path = "../output/Generated/" + discipline + "/"
    print(f"Verifying files in directory: {directory_path}")
    # Allowed prefixes for each new line
    allowed_prefixes = [ "user:", "assistant:"]
    question_count = 0
    def verify_file(file_path):
        global question_count
        with open(file_path, "r", encoding="utf-8") as file:
            inside_question_tag = False  # Flag to track if we are within <question> tags
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()
                
                # Check for <question> and </question> tags
                if "<question>" in stripped_line:
                    inside_question_tag = True
                    question_count += 1
                    continue
                if "</question>" in stripped_line:
                    inside_question_tag = False
                    continue
                
                # Skip lines within <question> tags
                if inside_question_tag:
                    continue
                
                # Check if the line starts with any of the allowed prefixes
                if stripped_line and not any(stripped_line.startswith(prefix) for prefix in allowed_prefixes):
                    print(f"Rule violated in file '{file_path}' at line {line_number}")
                    return False
        return True

    def verify_directory(directory_path):
        all_files_valid = True
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                if not verify_file(file_path):
                    all_files_valid = False
        return all_files_valid

    # Run the verification
    if verify_directory(directory_path):
        print("All files obey the rule!")
        print(f"Total number of questions: {question_count}")
    else:
        print("Some files violated the rule.")
        print(f"Total number of questions: {question_count}")
