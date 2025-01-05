import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# 设置显示选项，使输出的每一列对齐
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# 使用Pandas 打开' StudentInfo.csv '文件
df = pd.read_csv('StudentInfo.csv')

# 1. 查看data的统计信息
data_statistics = df.describe()
print("统计信息：")
print(data_statistics)

# 2. 处理空值，将平时成绩和期末成绩中的缺失值填充为0，每一列都可以单独拿来用
df.fillna({'平时成绩': 0}, inplace=True)
df.fillna({'期末成绩': 0}, inplace=True)

# 3. 处理重复值，去除学号这列的重复数据，保留重复行中的最后一行
df.drop_duplicates(subset=['学号'], keep='last', inplace=True)
print(df)

# 4. 处理异常值，检查期末成绩有没有异常值（<0 或者 >100)
out_of_range_values = df[(df['期末成绩'] < 0) | (df['期末成绩'] > 100)]
if not out_of_range_values.empty:
    print("异常值信息:")
    print(out_of_range_values)
    # 保留正确的值
    df = df[(df['期末成绩'] >= 0) & (df['期末成绩'] <= 100)]
else:
    print("没有异常值。")

# 5. 统计计算出期末成绩的平均计分、最大、最小、以及90分以上的人员个数和具体人员信息
average_score = df['期末成绩'].mean()
max_score = df['期末成绩'].max()
min_score = df['期末成绩'].min()
above_90_count = df[df['期末成绩'] > 90].shape[0]
above_90_info = df[df['期末成绩'] > 90]

print(f"平均成绩: {average_score}")
print(f"最高成绩: {max_score}")
print(f"最低成绩: {min_score}")
print(f"90分以上的人数: {above_90_count}")
print("90分以上的人员信息:")
print(above_90_info)

# 6. 按照'60分以下','60-70分','70-80分','80-90分','90-100分'统计成绩分布，并绘图显示
bins = [0, 60, 70, 80, 90, 100]
labels = ['60分以下', '60-70分', '70-80分', '80-90分', '90-100分']
df['成绩分布'] = pd.cut(df['期末成绩'], bins=bins, labels=labels, right=False)
score_distribution = df['成绩分布'].value_counts()

# 绘图
score_distribution.plot(kind='bar', color='red')
plt.title('成绩分布')
plt.xlabel('成绩区间')
plt.ylabel('人数')
plt.show()
