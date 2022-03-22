#pip install xlwt
#pip install openpyxl
import pandas as pd
path = 'pandas_to_excel.xlsx'
# df = pd.DataFrame([['aaa', 21.22, 31], [12, 22, 32], [31, 32, 33]],
#                   index=['one', 'two', 'three'], columns=['a', 'b', 'c'])

# print(df)
# df2 = df[['a', 'c']]
# print(df2)

# with pd.ExcelWriter(path) as writer:
#     df.to_excel(writer, sheet_name='Sheet1')
#     df2.to_excel(writer, sheet_name='Sheet2')


lst = [['type', 'img_name', 'label', 'pred_score', 'abs error']] 
x = ['train', '1.jpg', 1.88, 1.78, 0.1] 
lst.append(x) 
print(lst) 

df3 = pd.DataFrame(lst)
print(df3)
with pd.ExcelWriter(path) as writer:
    #df.to_excel(writer, sheet_name='Sheet1')
    df3.to_excel(writer, sheet_name='train', index=False)