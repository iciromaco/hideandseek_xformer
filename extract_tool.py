import os

def bundle_project_files(output_file="project_summary.txt"):
    # 解析対象とするファイルのリスト
    target_files = [
        "main28_train_final.py",
        "src/models/ppo_transformer_v2.py",
        "src/envs/hns28_environment.py",
        "src/envs/env_xml_builder.py",
        "configs/hparams_main28.toml",
        "src/core/obs_indices.py",
        "src/policy/policy_adapter.py",
        "src/agents/scripted_agents.py"
    ]

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("=== Project Structure and Source Code Summary ===\n")
        outfile.write(f"Generated for NotebookLM analysis\n\n")

        for file_path in target_files:
            if os.path.exists(file_path):
                outfile.write(f"\n{'='*20}\n")
                outfile.write(f"--- File: {file_path} ---\n")
                outfile.write(f"{'='*20}\n\n")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n")
                except Exception as e:
                    outfile.write(f"[Error reading file {file_path}: {e}]\n")
            else:
                print(f"Warning: {file_path} not found. Skipping...")

    print(f"Successfully created: {output_file}")

if __name__ == "__main__":
    bundle_project_files()