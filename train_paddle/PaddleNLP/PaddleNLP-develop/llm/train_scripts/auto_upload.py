import os
os.environ["AISTUDIO_ACCESS_TOKEN"] = "332948c58612d1d1790897f9220b69cf6a5c3600"

ckpt_list = [
    ('/root/ckpts/yulan_3_xpu_pt_fp32/', 'ZhipengChen/yulan3base_kunlun_p800_pretrain'),
    ('/root/ckpts/yulan3_base_xpu_sft/', 'ZhipengChen/yulan3base_kunlun_p800_sft'),
    ('/home/chenzhipeng/model/', 'ZhipengChen/yulan3base_kunlun_p800'),
]

from aistudio_sdk.hub import upload_file


for ckpt_folder, repo_id in ckpt_list:
    print('Uploading {}: {} ...'.format(repo_id, ckpt_folder))
    for file_name in os.listdir(ckpt_folder):
        file_path = os.path.join(ckpt_folder, file_name)
        print(file_path)
        if (os.path.isdir(file_path) == True):
            continue

        res = upload_file(
            repo_id=repo_id,
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            commit_message='upload model file to repo'
        )
        print(res)
