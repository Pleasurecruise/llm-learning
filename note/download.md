## Huggingface

```python
from transformers import AutoModel

model_name = "diffusers/t5-nf4"

model = AutoModel.from_pretrained(model_name, cache_dir='F:/huggingface')
```

## ModelScope

```python
from modelscope import snapshot_download

download_dir = 'f:/modelscope'

model_dir = snapshot_download('unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit', cache_dir=download_dir)

print(f'Model downloaded to: {model_dir}')
```

## ModelScope Dataset

