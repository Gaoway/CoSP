from datasets import load_dataset
import json
import random

datadir = "/data0/amax/cache/dataset/"

# ds  = load_dataset("allenai/swag", "regular", cache_dir= datadir)
# ds = load_dataset("Malikeh1375/medical-question-answering-datasets", "chatdoctor_icliniq", cache_dir= datadir)
ds = load_dataset("OpenAssistant/oasst2", cache_dir= datadir)

# train_data = ds['train']['input']
valid= ds['validation']['text']
filt = [x for x in valid if len(x) <= 255]
# filt = train_data.filter(lambda x: len(x) <= 255).to_dict()
# test_data = random.sample(train_data, 2000)
# test_data = test_data.to_dict()

with open( '/data0/amax/git/CoSP/dataset/'+'oasst.json', 'w', encoding='utf-8') as f:
    json.dump(filt, f, ensure_ascii=False, indent=4)