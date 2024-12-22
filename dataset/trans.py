import pandas as pd
import pyarrow.parquet as pq
import json

# 读取 Parquet 文件
parquet_file = "/data0/amax/git/CoSP/dataset/test-00000-of-00001.parquet"  # 替换为你的文件路径

try:
    # 尝试加载 Parquet 文件
    table = pq.read_table(parquet_file)
    print(table.schema)
except Exception as e:
    print(f"文件加载失败: {e}")
# df = pd.read_parquet(parquet_file)

# # 确保数据集有需要的字段，通常 WikiText 数据集的列可能是 "text" 等
# if "text" not in df.columns:
#     raise ValueError("Parquet 文件中没有 'text' 列，请检查数据结构！")

# custom_json = [row["text"] for _, row in df.iterrows()]

# json_file = "wikitext2.json"
# with open(json_file, "w", encoding="utf-8") as f:
#     json.dump(custom_json, f, ensure_ascii=False, indent=4)
# curl -X GET \
#      "https://datasets-server.huggingface.co/splits?dataset=mikasenghaas%2Fwikitext-2"
# curl -X GET \
#      "https://datasets-server.huggingface.co/rows?dataset=mikasenghaas%2Fwikitext-2&config=default&split=train&offset=0&length=100"