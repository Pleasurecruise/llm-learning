from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'bc09db5a-1ba9-4ef5-8fc7-6021f6687e0e'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'Pleasure1234'
model_name = 'wsj0-processed'
model_id = f"{owner_name}/{model_name}"

api.upload_folder(
    repo_id=f"{owner_name}/{model_name}",
    folder_path='C:\\Users\\31968\\Downloads\\Compressed\\wsj0',
    commit_message='upload dataset folder to repo',
    repo_type='dataset'
)