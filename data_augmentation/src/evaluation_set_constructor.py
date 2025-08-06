import pandas as pd

def main():
    path = "../dataset/data/"
    versions = ["attack", "idk", "attack_idk", "normal"]

    combined_df = pd.DataFrame()

    for version in versions:
        with open(f"{path}/processed/test_{version}.jsonl", "r", encoding="utf-8") as file:
            # read json file
            df = pd.read_json(file, lines=True)
        print(f"Processed {version} dataset with {len(df)} entries.")
        # filter entries where input contains more than 2 triple line breaks
        df = df[df.input.str.count("\n\n\n") < 2]
        # sample 250 entries from each dataset
        df_sampled = df.sample(n=250, random_state=42)
        if combined_df.empty:
            combined_df = df_sampled
        else:
            combined_df = pd.concat([combined_df, df_sampled], ignore_index=True)
    print(f"Combined dataset has {len(combined_df)} entries.")

    constrained_prompt = """You are a helpful chatbot that helps with users *solve academic problems*.\n\n
                            Your responses should be structured using a guided reasoning format enclosed within 
                            `<guidance>` and `</guidance>` tags."""

    # replace the full prompt in the first sample (split[0], when splitting by "\n\n\n") with the constrained prompt
    constrained_df = combined_df.copy()
    constrained_df.input = combined_df.input.apply(lambda x: x.replace(x.split("\n\n\n")[0], constrained_prompt))

    # save the constrained_df to a new file
    constrained_df.to_json(f"{path}/evaluation_constrained.jsonl", orient="records", lines=True)
    combined_df.to_json(f"{path}/evaluation.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    main()