# Generate stats about the generated dataset
import sys
import os
import statistics
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def calculate_stats(deciplines):
  for decipline in deciplines:
    # read every file in the output directory and count the number of <guidance> tags and calculate the average number of guidance tags and the average number of tokens in the file
    print("\n" + decipline + " Dataset")
    print("---------------------")
    dataset_path = "../output/" + decipline
    files = os.listdir(dataset_path)
    guidance_count = 0
    total_tokens = 0
    total_files = 0
    for file in files:
        if file.endswith(".md"):
            total_files += 1
            with open(os.path.join(dataset_path, file), "r") as f:
                data = f.read()
                guidance_count += data.count("<guidance>")
                total_tokens += len(data.split())
    if total_files > 0:
        avg_guidance = guidance_count / total_files
        avg_tokens = total_tokens / total_files

    print(f"Average number of guidance tags: {avg_guidance}")
    print(f"Total files generated: {total_files}")
    print(f"Average number of characters: {avg_tokens}")

    # calculate std deviation of the number of tokens in the files
    tokens = []
    for file in files:
        if file.endswith(".md"):
            with open(os.path.join(dataset_path, file), "r") as f:
                data = f.read()
                tokens.append(len(data.split()))
    try:
        std_dev = statistics.stdev(tokens)
    except statistics.StatisticsError:
        print("Not enough data to calculate standard deviation")
        std_dev = 0

    print(f"Standard deviation of the number of characters: {std_dev}")

    # calculate the std deviation of the number of guidance tags in the files

    guidance_tags = []
    for file in files:
        if file.endswith(".md"):
            with open(os.path.join(dataset_path, file), "r") as f:
                data = f.read()
                guidance_tags.append(data.count("<guidance>"))
    try:
        std_dev_guidance = statistics.stdev(guidance_tags)
    except statistics.StatisticsError:
        print("Not enough data to calculate standard deviation")
        std_dev_guidance = 0

    print(f"Standard deviation of the number of guidance tags: {std_dev_guidance}")

calculate_stats(["Economics", "Mathematics", "Biology", "Chemistry", "Statistics", "undisciplined", ])

# For first 2901 generated chain of thought text files
# Economics Dataset
# ---------------------
# Average number of guidance tags: 2.9624352331606216
# Total files generated: 1544
# Average number of characters: 392.45725388601034
# Standard deviation of the number of characters: 164.77362301959874
# Standard deviation of the number of guidance tags: 1.4040542268331682

# Mathematics Dataset
# ---------------------
# Average number of guidance tags: 3.0582959641255605
# Total files generated: 446
# Average number of characters: 351.6883408071749
# Standard deviation of the number of characters: 200.11049576532247
# Standard deviation of the number of guidance tags: 2.001395190289517

# Biology Dataset
# ---------------------
# Average number of guidance tags: 3.3743093922651934
# Total files generated: 724
# Average number of characters: 327.8660220994475
# Standard deviation of the number of characters: 160.43298090228996
# Standard deviation of the number of guidance tags: 1.4696383926368475

# Chemistry Dataset
# ---------------------
# Average number of guidance tags: 3.7291666666666665
# Total files generated: 48
# Average number of characters: 512.1041666666666
# Standard deviation of the number of characters: 195.7406272947925
# Standard deviation of the number of guidance tags: 1.52621884788231

# Statistics Dataset
# ---------------------
# Average number of guidance tags: 3.0592592592592593
# Total files generated: 135
# Average number of characters: 437.85185185185185
# Standard deviation of the number of characters: 134.50364909428808
# Standard deviation of the number of guidance tags: 1.2623300119812968

# undisciplined Dataset
# ---------------------
# Average number of guidance tags: 3.0
# Total files generated: 4
# Average number of characters: 324.5
# Standard deviation of the number of characters: 133.66251032606962
# Standard deviation of the number of guidance tags: 1.4142135623730951
