# 파이썬 기반의 머신러닝과 생태계의 이해
## 04. 데이터 핸들링 - 판다스

판다스의 핵심 객체: Dataframe

Index: 개별 데이터를 고유하게 식별하는 Key값(like RDBMS의 PK)

Series와 Dataframe은 모두 Index를 Key값으로 가짐

Series: 칼럼이 하나뿐인 데이터 구조체

Dataframe: 칼럼이 여러 개인 데이터 구조체(여러 개의 Series로 이루어짐)
* * *

### 판다스 시작 - 파일을 Dataframe으로 로딩, 기본 API


```python
import pandas as pd
```

판다스는 다양한 포맷으로 된 파일을 Dataframe으로 로딩할 수 있는 API 제공

* read_csv() : 필드 구분 문자가 콤마(',') 
* read_table() : 필드 구분 문자가 탭('\t')
* read_fwf() : 고정 길이 기반의 칼럼 포맷을 Dataframe으로 로딩하기 위한 API

필드 구분 문자를 변경하고 싶다면 read_csv('filename',sep='\t')와 같이 변경할 수 있음


```python
titanic_df = pd.read_csv('titanic_train.csv')
print('titanic 변수 type:', type(titanic_df))
titanic_df
```

    titanic 변수 type: <class 'pandas.core.frame.DataFrame'>
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>



* shape 인자 : 행과 열 크기 투플 형태로 반환
* head() : 맨 앞에 있는 N개의 로우 반환
* info() : 총 데이터 건수, 데이터 타입, Null건수 등의 정보 반환
* describe() : 칼럼별 숫자형 데이터값의 n-percentile 분포도, 평균값, 최댓값, 최솟값(int, float등의 숫자형만 조사)


```python
print(titanic_df.shape)
titanic_df.info()
titanic_df.describe()
```

    (891, 12)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



데이터프레임의 [] 내부에 칼럼명 입력하면 Series 형태로 특정 칼럼 데이터 세트 반환

* value_counts() : 지정된 칼럼의 데이터값 건수 반환 -> 데이터의 분포도를 확인하는 데 매우 유용
 
칼럼 값별 데이터 건수 반환 -> 고유 칼럼 값을 식별자로 사용 가능

인덱스의 고유성이 보장된다면 의미있는 데이터값 할당 가능


```python
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
```

    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64
    

* * *

### DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호 변환

Dataframe은 기본적으로 행과 열을 가지는 2차원 데이터이므로 2차원 이하의 데이터들만 Dataframe으로 변환 가능



```python
import numpy as np

col_name1 = ['col1']
list1 = [1,2,3]
array1 = np.array(list1)
print('array1 shape:', array1.shape)

# 리스트를 이용한 데이터프레임 생성
df_list1 = pd.DataFrame(list1,columns=col_name1)
print('1차원 리스트로 만든 데이터프레임:\n', df_list1)

# 넘파이 ndarray를 이용한 데이터프레임 생성
df_array1 = pd.DataFrame(array1,columns=col_name1)
print('1차원 ndarray로 만든 데이터프레임:\n',df_array1)
```

    array1 shape: (3,)
    1차원 리스트로 만든 데이터프레임:
        col1
    0     1
    1     2
    2     3
    1차원 ndarray로 만든 데이터프레임:
        col1
    0     1
    1     2
    2     3
    

딕셔너리를 데이터프레임으로 변환 시

Key는 문자열 칼럼명으로 매핑, Value는 리스트 형(또는 ndarray) 칼럼 데이터로 매핑

-> 키의 경우는 문자열, 값의 경우 리스트(또는 ndarray) 형태로 딕셔너리 구성


```python
dict = {'col1':[1,11],'col2':[2,22],'col3':[3,33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 데이터프레임:\n', df_dict)
```

    딕셔너리로 만든 데이터프레임:
        col1  col2  col3
    0     1     2     3
    1    11    22    33
    

* * *

### DataFrame을 넘파이 ndarray, 리스트, 딕셔너리로 변환하기



```python
# 데이터프레임을 ndarray로 변환
array3 = df_dict.values
print('df_dict.values 타입:',type(array3),'\ndf_dict.values shape:',array3.shape)
print(array3)

# 데이터프레임을 리스트로 변환
list3 = df_dict.values.tolist()
print('\ndf_dict.values.tolist() 타입:', type(list3))
print(list3)

# 데이터프레임을 딕셔너리로 변환
dict3 = df_dict.to_dict('list')
print('\ndf_dict.to_dict() 타입:', type(dict3))
print(dict3)
```

    df_dict.values 타입: <class 'numpy.ndarray'> 
    df_dict.values shape: (2, 3)
    [[ 1  2  3]
     [11 22 33]]
    
    df_dict.values.tolist() 타입: <class 'list'>
    [[1, 2, 3], [11, 22, 33]]
    
    df_dict.to_dict() 타입: <class 'dict'>
    {'col1': [1, 11], 'col2': [2, 22], 'col3': [3, 33]}
    

* * *

### DataFrame의 칼럼 데이터 세트 생성과 수정

DataFrame[ ] 내에 새로운 칼럼명 입력하고 값을 할당

기존 칼럼의 데이터를 이용해 새로운 칼럼 생성도 가능


```python
titanic_df['Age_0']=0
titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



* * *

### DataFrame 데이터 삭제

* drop() : 데이터의 삭제

> 원형 : DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
> > * axis : 이 값에 따라서 특정 칼럼 또는 특정 행을 드롭(0은 로우 방향 축, 1은 칼럼 방향 축)
> > * labels : 원하는 로우/칼럼 명
> > * inplace : True이면 원본 데이터프레임에 영향, False이면 영향 없이 객체변수만 영향
> >             주의! inplace가 True일 때 반환값이 None이므로 이를 다시 데이터프레임에 할당하면 안됨


```python
titanic_drop_df = titanic_df.drop('Age_0', axis=1)
titanic_drop_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



* * *

### Index 객체

판다스의 인덱스 객체 : DataFrame, Series의 레코드를 고유하게 식별하는 객체

인덱스 추출 : Series.index 혹은 DataFrame.index 속성


```python
# 원본 파일 다시 로딩
titanic_df = pd.read_csv('titanic_train.csv')
# Index 객체 추출
indexes = titanic_df.index
print(indexes)
# Index 객체를 실제 값 array로 변환
print('\nIndex 객체 array 값:\n', indexes.values)
```

    RangeIndex(start=0, stop=891, step=1)
    
    Index 객체 array 값:
     [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
      18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
      36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
      54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
      72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
      90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
     108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
     126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
     144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
     162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
     180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197
     198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215
     216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233
     234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251
     252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269
     270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287
     288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305
     306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323
     324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341
     342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359
     360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377
     378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395
     396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413
     414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431
     432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449
     450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467
     468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485
     486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503
     504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521
     522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539
     540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557
     558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575
     576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593
     594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611
     612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629
     630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647
     648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665
     666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683
     684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701
     702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719
     720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737
     738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755
     756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773
     774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791
     792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809
     810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827
     828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845
     846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863
     864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881
     882 883 884 885 886 887 888 889 890]
    

* * *

### 데이터 셀렉션 및 필터링

넘파이의 [ ] 연산자 내 단일 값 추출, 슬라이싱, 팬시 인덱싱, 불린 인덱싱

판다스의 경우 ix[ ], iloc[ ], loc[ ] 연산자를 통해 동일한 작업 수행

* ix[] : 명칭 기반 인덱싱, 위치 기반 인덱싱 모두 적용 가능
* loc[] : 명칭 기반 인덱싱, loc['one','name']와 같이 명칭 지정으로 원하는 위치의 데이터 추출, 슬라이싱 시 종료점 포함
* iloc[] : 위치 기반 인덱싱, iloc[0,3]과 같이 위치 지정으로 원하는 위치의 데이터 추출

* * *

### 정렬, Aggregation 함수, GroupBy 적용

* sort_values() : DataFrame과 Series의 정렬

    by : 특정 칼럼 입력하면 해당 칼럼으로 정렬 수행
    ascending : True이면 오름차순, False이면 내림차순, True가 기본
    inplace : True이면 호출한 데이터프레임의 정렬 결과 그대로 적용, False이면 적용 안함, False가 기본


```python
titanic_sorted = titanic_df.sort_values(by=['Name','Pclass'], ascending=False)
titanic_sorted.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>868</th>
      <td>869</td>
      <td>0</td>
      <td>3</td>
      <td>van Melkebeke, Mr. Philemon</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>345777</td>
      <td>9.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>153</th>
      <td>154</td>
      <td>0</td>
      <td>3</td>
      <td>van Billiard, Mr. Austin Blyler</td>
      <td>male</td>
      <td>40.5</td>
      <td>0</td>
      <td>2</td>
      <td>A/5. 851</td>
      <td>14.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>361</th>
      <td>362</td>
      <td>0</td>
      <td>2</td>
      <td>del Carlo, Mr. Sebastiano</td>
      <td>male</td>
      <td>29.0</td>
      <td>1</td>
      <td>0</td>
      <td>SC/PARIS 2167</td>
      <td>27.7208</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(titanic_df.count())
titanic_df[['Age','Fare']].mean()
```

    PassengerId    891
    Survived       891
    Pclass         891
    Name           891
    Sex            891
    Age            714
    SibSp          891
    Parch          891
    Ticket         891
    Fare           891
    Cabin          204
    Embarked       889
    dtype: int64
    




    Age     29.699118
    Fare    32.204208
    dtype: float64



* groupby() : 입력 파라미터 by에 칼럼을 입력하면 대상 칼럼으로 groupby됨
* DataFrame에 groupby() 호출하면 DataFrameGroupBy라는 또 다른 형태의 데이터프레임 반환


```python
# Pclass칼럼 기준으로 groupby된 객체 반환
titanic_groupby = titanic_df.groupby(by='Pclass') 
print(type(titanic_groupby))

# groupby()를 호출해 반환된 결과에 Aggregation함수를 적용
# 대상 칼럼 제외한 모든 칼럼에 함수 적용
titanic_groupby = titanic_df.groupby(by='Pclass').count()
titanic_groupby

# 필터링 해서 수행
titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId', 'Survived']].count()
titanic_groupby
```

    <class 'pandas.core.groupby.generic.DataFrameGroupBy'>
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>216</td>
      <td>216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>184</td>
      <td>184</td>
    </tr>
    <tr>
      <th>3</th>
      <td>491</td>
      <td>491</td>
    </tr>
  </tbody>
</table>
</div>



* * *

### 결손 데이터 처리하기

결손 데이터 : 칼럼에 값이 없는, 즉 NULL인 경우를 의미, 넘파이의 NaN으로 표시

* isna() : NaN 여부 확인
* fillna() : NaN 값을 다른 값으로 대체
             주의! 반환 값을 다시 받거나 inplace=True이어야지만 실제 데이터셋 값이 변경됨


```python
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()
```




    PassengerId    0
    Survived       0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Cabin          0
    Embarked       0
    dtype: int64



* * * 

### apply lambda 식으로 데이터 가공




```python

```
