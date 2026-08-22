from huggingface_hub import list_repo_tree

for entry in list_repo_tree("ai4bharat/MSMARCO-XI", repo_type="dataset", recursive=True):
    print(entry.path)
