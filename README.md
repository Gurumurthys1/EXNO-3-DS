## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data (1).csv")
df
```
![Screenshot 2024-04-02 144335](https://github.com/23007232/EXNO-3-DS/assets/139115574/9d085fc9-4fe5-4159-9305-e382888823d9)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/b4aaab8c-4b40-4131-8034-37a5f091c24f)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/cebcb8ac-625c-48f3-a9ba-7c9f3036a532)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/0068ebc8-fc93-4bfd-acea-bd98d0f896cb)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/2e02e687-8d9d-48f4-b1ec-89d3ab3a9092)
```
pd.get_dummies(df2,columns=["nom_0"])
```

![Screenshot 2024-04-02 152247](https://github.com/23007232/EXNO-3-DS/assets/139115574/303b629a-9d26-4362-9208-792a56075e91)

```
import pandas as pd
pip install --upgrade category_encoders
```
![Screenshot 2024-04-02 152446](https://github.com/23007232/EXNO-3-DS/assets/139115574/421fd860-6397-4468-8d90-40662ba19808)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data (1).csv")
df
```
![Screenshot 2024-04-02 152614](https://github.com/23007232/EXNO-3-DS/assets/139115574/3257608d-4590-442f-a9e3-388cfb181cae)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb=df.copy()
dfb
```

![Screenshot 2024-04-02 152748](https://github.com/23007232/EXNO-3-DS/assets/139115574/82c2a279-234d-4510-9da1-c385b2354b11)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/50b6ba1a-6b27-4ef4-b97a-200cacf262db)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/30b1b5da-5245-45cb-9770-1709c66e9010)
```
df.skew()
np.log(df["Highly Positive Skew"])
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/69976ed0-431d-4c7a-94ee-37d9e6da055e)
```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/f283d52a-20c3-4e15-bf21-17a5629632f9)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/035fb8b3-6fca-4072-b26e-a9d078977114)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/3809b8c3-3e1c-40d2-94b0-8510141d6f0e)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/4a247bd7-4edf-4184-89ac-04993b38d169)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/c7a3d3d4-87d4-4ec0-a3cc-058bd7ee3049)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/58a008f5-feea-48e4-804a-7ba4dad00211)

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/c7b7023f-5e3e-42df-a8b8-2da2590e2469)

![image](https://github.com/23007232/EXNO-3-DS/assets/139115574/ae3e267c-c9b1-41e5-ab07-7500853c8ba4)


# RESULT:
Thus,the Feature Encoding and Transformation process has been performed on the given data.


       
