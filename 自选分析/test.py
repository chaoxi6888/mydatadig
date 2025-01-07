import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 读取 CSV 文件，指定编码
file_path = 'meal (2).csv'  # 文件路径
df = pd.read_csv(file_path, header=0)  # 读取 CSV 文件并指定编码

# 将 DataFrame 转换为布尔类型
df = df.astype(bool)

# 使用 Apriori 算法找出频繁项集
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# 生成关联规则
rules = association_rules(frequent_itemsets, frequent_itemsets, metric="confidence", min_threshold=0.5)

# 打印频繁项集和关联规则
print("频繁项集：")
print(frequent_itemsets)
print("\n关联规则：")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
