from huggingface_hub import hf_hub_download

path = hf_hub_download("ai4bharat/MSMARCO-XI", "ms_marco_translations.py", repo_type="dataset")
print(path)
print(open(path, encoding="utf-8").read())
