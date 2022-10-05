# XGBoost

: 트리 기반의 앙상블 학습 중에서 가장 각광받고 있는 알고리즘 중 하나
  
  GBM에 기반하고 있지만 그 단점(느린 수행 시간, 과적합 규제의 부재 등)을 해결하여 매우 각광
  
  병렬 CPU환경에서 병렬 학습이 가능해 기존 GBM보다 빠르게 학습을 완료할 수 있음
  
* XGBoost의 핵심 라이브러리는 C/C++로 작성되어 있음, 파이썬에서도 구동할 수 있도록 파이썬 패키지 제공(C/C++핵심 라이브러리 호출)

  파이썬 패키지는 초기에는 사이킷런과 호환되지 않는 독자적인 XGBoost 전용 패키지였음, 따라서 사이킷런 고유의 아키텍처 적용이 불가능했음

  따라서 사이킷런과 연동할 수 있는 래퍼 클래스(Wrapper class) 제공 - XGBClassifier, XGBRegreesor)
  
  
* 파이썬 래퍼 XGBoost 모듈 : 초기의 독자적인 XGBoost 프레임워크 기반의 XGBoost
* 사이킷런 래퍼 XGBoost 모듈 : 사이킷런과 연동되는 모듈


```python
import xgboost as xgb
from xgboost import XGBClassifier
```

    C:\Users\gram\anaconda3\lib\site-packages\xgboost\compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
      from pandas import MultiIndex, Int64Index
    


```python
import xgboost

print(xgboost.__version__)
```

    1.5.0
    

> XGBoost에서 과적합 문제가 심각하다면 고려해야 할 것
> 
> * eta 값을 낮춤(0.01~0.1). eta 값을 낮출 경우 num_round(또는 n_estimators)는 반대로 높여줘야 함
> * max_depth 값을 낮춤
> * min_child_weight 값을 높임
> * gamma 값을 높임
> * subsample과 colsample_bytree를 조정하는 것도 트리가 너무 복잡하게 생성되는 것을 막아 과적합 문제에 도움이 될 수 있음

* 조기 중단(Early Stopping) : n_estimator에서 지정한 부스팅 반복 횟수에 도달하지 않더라도 예측 오류가 더이상 개선되지 않으면 중지해 수행 시간 개선

* * *

## 파이썬 래퍼 XGBoost 적용 - 위스콘신 유방암 예측

> 위스콘신 유방암 데이터
> : 종양의 크기, 모양 등의 다양한 속성값을 기반으로 악성 종양인지 양성 종양인지 분류한 데이터셋


```python
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
X_features = dataset.data
y_label = dataset.target

cancer_df = pd.DataFrame(data=X_features, columns=dataset.feature_names)
cancer_df['target']=y_label
cancer_df.head(3)
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.8</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.6</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.9</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.8</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.0</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.5</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
</div>



> 위의 결과 분석
> * 종양의 크기와 모양에 관련된 많은 속성이 숫자형 값으로 되어 있음
> * 타깃 레이블 값의 종류는 악성(malignant)가 0, 양성(benign)이 1로 되어있음


```python
# 레이블 값의 분포 확인
print(dataset.target_names)
print(cancer_df['target'].value_counts())
```

    ['malignant' 'benign']
    1    357
    0    212
    Name: target, dtype: int64
    


```python
# 전체 데이터 중 80%는 학습용, 20%는 테스트용 데이터 추출
X_train, X_test, y_train, y_test=train_test_split(X_features, y_label, test_size=0.2, random_state=156)
print(X_train.shape, X_test.shape)
```

    (455, 30) (114, 30)
    

파이썬 래퍼 XGBoost는 학습용과 테스트용 데이터셋을 위해 별도의 객체인 DMatrix를 생성

DMatrix는 주로 넘파이 입력 파라미터를 받아서 만들어지는 XGBoost만의 전용 데이터셋

DMatrix의 주요 입력 파라미터 : ( data: 피처 데이터셋, label: 레이블 데이터셋(분류)/숫자형 종속값 데이터셋(회귀) )


DMatrix는 넘파이 외에 libsvm txt 포맷 파일, xgboost 이진 버퍼 파일을 파라미터로 입력받아 변환 가능

판다스의 DataFrame으로 데이터 인터페이스를 하기 위해서는 DataFrame.values를 이용해 넘파이로 일차변환 후 DMatrix변환 적용


```python
# 넘파이 형태의 학습 데이터셋과 테스트 데이터셋을 DMatrix로 변환하는 예제
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)
```

* 파이썬 래퍼 XGBoost 모듈인 xgboost 이용하여 학습 수행

> XGBoost 하이퍼 파라미터 설정 (주로 딕셔너리 형태로 입력)
> * max_depth는 3
> * 학습률 eta는 0.1(XGBClassifier 사용할 경우 eta가 아니라 learning_rate)
> * 데이터가 0또는1 이진 분류이므로 목적함수(objective)는 이진 로지스틱(binary:logistic)
> * 오류 함수의 평가 성능 지표는 logloss
> * num_rounds(부스팅 반복 횟수)는 400회

> eary_stopping_rounds 파라미터를 설정해 조기 중단을 수행하기 위해서는 반드시 eval_set과 eval_metric이 함께 설정되어야 함
> XGBoost는 반복마다 eval_set로 지정된 데이터 세트에서 eval_metric의 지정된 평가 지표로 예측 오류 측정
> * eval_set : 성능 평가를 수행할 평가용 데이터셋 설정
> * eval_metric : 평가 세트에 적용할 성능 평가 방법. 분류의 경우 주로 'error'(분류 오류), 'logloss' 적용


```python
params = {
    'max_depth':3,
    'eta':0.1,
    'objective':'binary:logistic',
    'eval_metric':'logloss',
    'early_stoppings':100
    }
num_rounds = 400
```


```python
# train 데이터 세트는 'train', evaluation(test) 데이터 세트는 'eval'로 명기
wlist = [(dtrain, 'train'), (dtest, 'eval')]
# 하이퍼 파라미터와 early stopping 파라미터를 train()함수의 파라미터로 전달
xgb_model = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds, early_stopping_rounds=100, evals=wlist)
```

    [19:15:31] WARNING: ..\src\learner.cc:576: 
    Parameters: { "early_stoppings" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
    [0]	train-logloss:0.60969	eval-logloss:0.61352
    [1]	train-logloss:0.54080	eval-logloss:0.54784
    [2]	train-logloss:0.48375	eval-logloss:0.49425
    [3]	train-logloss:0.43446	eval-logloss:0.44799
    [4]	train-logloss:0.39055	eval-logloss:0.40911
    [5]	train-logloss:0.35414	eval-logloss:0.37498
    [6]	train-logloss:0.32122	eval-logloss:0.34571
    [7]	train-logloss:0.29259	eval-logloss:0.32053
    [8]	train-logloss:0.26747	eval-logloss:0.29721
    [9]	train-logloss:0.24515	eval-logloss:0.27799
    [10]	train-logloss:0.22569	eval-logloss:0.26030
    [11]	train-logloss:0.20794	eval-logloss:0.24604
    [12]	train-logloss:0.19218	eval-logloss:0.23156
    [13]	train-logloss:0.17792	eval-logloss:0.22005
    [14]	train-logloss:0.16522	eval-logloss:0.20857
    [15]	train-logloss:0.15362	eval-logloss:0.19999
    [16]	train-logloss:0.14333	eval-logloss:0.19012
    [17]	train-logloss:0.13398	eval-logloss:0.18182
    [18]	train-logloss:0.12560	eval-logloss:0.17473
    [19]	train-logloss:0.11729	eval-logloss:0.16766
    [20]	train-logloss:0.10969	eval-logloss:0.15820
    [21]	train-logloss:0.10297	eval-logloss:0.15473
    [22]	train-logloss:0.09707	eval-logloss:0.14895
    [23]	train-logloss:0.09143	eval-logloss:0.14331
    [24]	train-logloss:0.08633	eval-logloss:0.13634
    [25]	train-logloss:0.08131	eval-logloss:0.13278
    [26]	train-logloss:0.07686	eval-logloss:0.12791
    [27]	train-logloss:0.07284	eval-logloss:0.12526
    [28]	train-logloss:0.06925	eval-logloss:0.11998
    [29]	train-logloss:0.06555	eval-logloss:0.11641
    [30]	train-logloss:0.06241	eval-logloss:0.11450
    [31]	train-logloss:0.05959	eval-logloss:0.11257
    [32]	train-logloss:0.05710	eval-logloss:0.11154
    [33]	train-logloss:0.05441	eval-logloss:0.10868
    [34]	train-logloss:0.05204	eval-logloss:0.10668
    [35]	train-logloss:0.04975	eval-logloss:0.10421
    [36]	train-logloss:0.04775	eval-logloss:0.10296
    [37]	train-logloss:0.04585	eval-logloss:0.10058
    [38]	train-logloss:0.04401	eval-logloss:0.09868
    [39]	train-logloss:0.04226	eval-logloss:0.09644
    [40]	train-logloss:0.04065	eval-logloss:0.09587
    [41]	train-logloss:0.03913	eval-logloss:0.09424
    [42]	train-logloss:0.03738	eval-logloss:0.09471
    [43]	train-logloss:0.03611	eval-logloss:0.09427
    [44]	train-logloss:0.03494	eval-logloss:0.09389
    [45]	train-logloss:0.03365	eval-logloss:0.09418
    [46]	train-logloss:0.03253	eval-logloss:0.09402
    [47]	train-logloss:0.03148	eval-logloss:0.09236
    [48]	train-logloss:0.03039	eval-logloss:0.09301
    [49]	train-logloss:0.02947	eval-logloss:0.09127
    [50]	train-logloss:0.02855	eval-logloss:0.09005
    [51]	train-logloss:0.02753	eval-logloss:0.08961
    [52]	train-logloss:0.02655	eval-logloss:0.08958
    [53]	train-logloss:0.02568	eval-logloss:0.09070
    [54]	train-logloss:0.02500	eval-logloss:0.08958
    [55]	train-logloss:0.02430	eval-logloss:0.09036
    [56]	train-logloss:0.02357	eval-logloss:0.09159
    [57]	train-logloss:0.02296	eval-logloss:0.09153
    [58]	train-logloss:0.02249	eval-logloss:0.09199
    [59]	train-logloss:0.02185	eval-logloss:0.09195
    [60]	train-logloss:0.02132	eval-logloss:0.09194
    [61]	train-logloss:0.02079	eval-logloss:0.09146
    [62]	train-logloss:0.02022	eval-logloss:0.09031
    [63]	train-logloss:0.01970	eval-logloss:0.08941
    [64]	train-logloss:0.01918	eval-logloss:0.08972
    [65]	train-logloss:0.01872	eval-logloss:0.08974
    [66]	train-logloss:0.01833	eval-logloss:0.08962
    [67]	train-logloss:0.01787	eval-logloss:0.08873
    [68]	train-logloss:0.01760	eval-logloss:0.08862
    [69]	train-logloss:0.01724	eval-logloss:0.08974
    [70]	train-logloss:0.01688	eval-logloss:0.08998
    [71]	train-logloss:0.01664	eval-logloss:0.08978
    [72]	train-logloss:0.01629	eval-logloss:0.08958
    [73]	train-logloss:0.01598	eval-logloss:0.08953
    [74]	train-logloss:0.01566	eval-logloss:0.08875
    [75]	train-logloss:0.01539	eval-logloss:0.08860
    [76]	train-logloss:0.01515	eval-logloss:0.08812
    [77]	train-logloss:0.01488	eval-logloss:0.08840
    [78]	train-logloss:0.01464	eval-logloss:0.08874
    [79]	train-logloss:0.01449	eval-logloss:0.08815
    [80]	train-logloss:0.01418	eval-logloss:0.08758
    [81]	train-logloss:0.01401	eval-logloss:0.08741
    [82]	train-logloss:0.01377	eval-logloss:0.08849
    [83]	train-logloss:0.01357	eval-logloss:0.08858
    [84]	train-logloss:0.01341	eval-logloss:0.08807
    [85]	train-logloss:0.01325	eval-logloss:0.08764
    [86]	train-logloss:0.01311	eval-logloss:0.08742
    [87]	train-logloss:0.01293	eval-logloss:0.08761
    [88]	train-logloss:0.01271	eval-logloss:0.08707
    [89]	train-logloss:0.01254	eval-logloss:0.08727
    [90]	train-logloss:0.01235	eval-logloss:0.08716
    [91]	train-logloss:0.01223	eval-logloss:0.08696
    [92]	train-logloss:0.01206	eval-logloss:0.08717
    [93]	train-logloss:0.01193	eval-logloss:0.08707
    [94]	train-logloss:0.01182	eval-logloss:0.08659
    [95]	train-logloss:0.01165	eval-logloss:0.08612
    [96]	train-logloss:0.01148	eval-logloss:0.08714
    [97]	train-logloss:0.01136	eval-logloss:0.08677
    [98]	train-logloss:0.01124	eval-logloss:0.08669
    [99]	train-logloss:0.01113	eval-logloss:0.08655
    [100]	train-logloss:0.01100	eval-logloss:0.08650
    [101]	train-logloss:0.01085	eval-logloss:0.08641
    [102]	train-logloss:0.01076	eval-logloss:0.08629
    [103]	train-logloss:0.01064	eval-logloss:0.08626
    [104]	train-logloss:0.01050	eval-logloss:0.08683
    [105]	train-logloss:0.01039	eval-logloss:0.08677
    [106]	train-logloss:0.01030	eval-logloss:0.08732
    [107]	train-logloss:0.01020	eval-logloss:0.08730
    [108]	train-logloss:0.01007	eval-logloss:0.08728
    [109]	train-logloss:0.01000	eval-logloss:0.08730
    [110]	train-logloss:0.00991	eval-logloss:0.08729
    [111]	train-logloss:0.00980	eval-logloss:0.08800
    [112]	train-logloss:0.00971	eval-logloss:0.08794
    [113]	train-logloss:0.00963	eval-logloss:0.08784
    [114]	train-logloss:0.00956	eval-logloss:0.08807
    [115]	train-logloss:0.00948	eval-logloss:0.08765
    [116]	train-logloss:0.00942	eval-logloss:0.08730
    [117]	train-logloss:0.00931	eval-logloss:0.08780
    [118]	train-logloss:0.00923	eval-logloss:0.08775
    [119]	train-logloss:0.00915	eval-logloss:0.08768
    [120]	train-logloss:0.00912	eval-logloss:0.08763
    [121]	train-logloss:0.00902	eval-logloss:0.08757
    [122]	train-logloss:0.00897	eval-logloss:0.08755
    [123]	train-logloss:0.00890	eval-logloss:0.08716
    [124]	train-logloss:0.00884	eval-logloss:0.08767
    [125]	train-logloss:0.00880	eval-logloss:0.08774
    [126]	train-logloss:0.00871	eval-logloss:0.08828
    [127]	train-logloss:0.00864	eval-logloss:0.08831
    [128]	train-logloss:0.00861	eval-logloss:0.08827
    [129]	train-logloss:0.00856	eval-logloss:0.08789
    [130]	train-logloss:0.00846	eval-logloss:0.08886
    [131]	train-logloss:0.00842	eval-logloss:0.08868
    [132]	train-logloss:0.00839	eval-logloss:0.08874
    [133]	train-logloss:0.00830	eval-logloss:0.08922
    [134]	train-logloss:0.00827	eval-logloss:0.08918
    [135]	train-logloss:0.00822	eval-logloss:0.08882
    [136]	train-logloss:0.00816	eval-logloss:0.08851
    [137]	train-logloss:0.00808	eval-logloss:0.08848
    [138]	train-logloss:0.00805	eval-logloss:0.08839
    [139]	train-logloss:0.00797	eval-logloss:0.08915
    [140]	train-logloss:0.00795	eval-logloss:0.08911
    [141]	train-logloss:0.00790	eval-logloss:0.08876
    [142]	train-logloss:0.00787	eval-logloss:0.08868
    [143]	train-logloss:0.00785	eval-logloss:0.08839
    [144]	train-logloss:0.00778	eval-logloss:0.08927
    [145]	train-logloss:0.00775	eval-logloss:0.08924
    [146]	train-logloss:0.00773	eval-logloss:0.08914
    [147]	train-logloss:0.00769	eval-logloss:0.08891
    [148]	train-logloss:0.00762	eval-logloss:0.08942
    [149]	train-logloss:0.00760	eval-logloss:0.08939
    [150]	train-logloss:0.00758	eval-logloss:0.08911
    [151]	train-logloss:0.00752	eval-logloss:0.08873
    [152]	train-logloss:0.00750	eval-logloss:0.08872
    [153]	train-logloss:0.00746	eval-logloss:0.08848
    [154]	train-logloss:0.00741	eval-logloss:0.08847
    [155]	train-logloss:0.00739	eval-logloss:0.08854
    [156]	train-logloss:0.00737	eval-logloss:0.08852
    [157]	train-logloss:0.00734	eval-logloss:0.08855
    [158]	train-logloss:0.00732	eval-logloss:0.08828
    [159]	train-logloss:0.00730	eval-logloss:0.08830
    [160]	train-logloss:0.00728	eval-logloss:0.08828
    [161]	train-logloss:0.00726	eval-logloss:0.08801
    [162]	train-logloss:0.00724	eval-logloss:0.08776
    [163]	train-logloss:0.00722	eval-logloss:0.08778
    [164]	train-logloss:0.00720	eval-logloss:0.08778
    [165]	train-logloss:0.00718	eval-logloss:0.08752
    [166]	train-logloss:0.00716	eval-logloss:0.08754
    [167]	train-logloss:0.00714	eval-logloss:0.08764
    [168]	train-logloss:0.00712	eval-logloss:0.08739
    [169]	train-logloss:0.00710	eval-logloss:0.08738
    [170]	train-logloss:0.00708	eval-logloss:0.08730
    [171]	train-logloss:0.00707	eval-logloss:0.08737
    [172]	train-logloss:0.00705	eval-logloss:0.08740
    [173]	train-logloss:0.00703	eval-logloss:0.08739
    [174]	train-logloss:0.00701	eval-logloss:0.08713
    [175]	train-logloss:0.00699	eval-logloss:0.08716
    [176]	train-logloss:0.00697	eval-logloss:0.08696
    [177]	train-logloss:0.00696	eval-logloss:0.08705
    [178]	train-logloss:0.00694	eval-logloss:0.08697
    [179]	train-logloss:0.00692	eval-logloss:0.08697
    [180]	train-logloss:0.00690	eval-logloss:0.08704
    [181]	train-logloss:0.00688	eval-logloss:0.08680
    [182]	train-logloss:0.00687	eval-logloss:0.08683
    [183]	train-logloss:0.00685	eval-logloss:0.08658
    [184]	train-logloss:0.00683	eval-logloss:0.08659
    [185]	train-logloss:0.00681	eval-logloss:0.08661
    [186]	train-logloss:0.00680	eval-logloss:0.08637
    [187]	train-logloss:0.00678	eval-logloss:0.08637
    [188]	train-logloss:0.00676	eval-logloss:0.08630
    [189]	train-logloss:0.00675	eval-logloss:0.08610
    [190]	train-logloss:0.00673	eval-logloss:0.08602
    [191]	train-logloss:0.00671	eval-logloss:0.08605
    [192]	train-logloss:0.00670	eval-logloss:0.08615
    [193]	train-logloss:0.00668	eval-logloss:0.08592
    [194]	train-logloss:0.00667	eval-logloss:0.08592
    [195]	train-logloss:0.00665	eval-logloss:0.08598
    [196]	train-logloss:0.00663	eval-logloss:0.08601
    [197]	train-logloss:0.00662	eval-logloss:0.08592
    [198]	train-logloss:0.00660	eval-logloss:0.08585
    [199]	train-logloss:0.00659	eval-logloss:0.08587
    [200]	train-logloss:0.00657	eval-logloss:0.08589
    [201]	train-logloss:0.00656	eval-logloss:0.08595
    [202]	train-logloss:0.00654	eval-logloss:0.08573
    [203]	train-logloss:0.00653	eval-logloss:0.08573
    [204]	train-logloss:0.00651	eval-logloss:0.08575
    [205]	train-logloss:0.00650	eval-logloss:0.08582
    [206]	train-logloss:0.00648	eval-logloss:0.08584
    [207]	train-logloss:0.00647	eval-logloss:0.08578
    [208]	train-logloss:0.00645	eval-logloss:0.08569
    [209]	train-logloss:0.00644	eval-logloss:0.08571
    [210]	train-logloss:0.00643	eval-logloss:0.08581
    [211]	train-logloss:0.00641	eval-logloss:0.08559
    [212]	train-logloss:0.00640	eval-logloss:0.08580
    [213]	train-logloss:0.00639	eval-logloss:0.08581
    [214]	train-logloss:0.00637	eval-logloss:0.08574
    [215]	train-logloss:0.00636	eval-logloss:0.08566
    [216]	train-logloss:0.00634	eval-logloss:0.08584
    [217]	train-logloss:0.00633	eval-logloss:0.08563
    [218]	train-logloss:0.00632	eval-logloss:0.08573
    [219]	train-logloss:0.00631	eval-logloss:0.08578
    [220]	train-logloss:0.00629	eval-logloss:0.08579
    [221]	train-logloss:0.00628	eval-logloss:0.08582
    [222]	train-logloss:0.00627	eval-logloss:0.08576
    [223]	train-logloss:0.00626	eval-logloss:0.08567
    [224]	train-logloss:0.00624	eval-logloss:0.08586
    [225]	train-logloss:0.00623	eval-logloss:0.08587
    [226]	train-logloss:0.00622	eval-logloss:0.08593
    [227]	train-logloss:0.00621	eval-logloss:0.08595
    [228]	train-logloss:0.00619	eval-logloss:0.08587
    [229]	train-logloss:0.00618	eval-logloss:0.08606
    [230]	train-logloss:0.00617	eval-logloss:0.08600
    [231]	train-logloss:0.00616	eval-logloss:0.08592
    [232]	train-logloss:0.00615	eval-logloss:0.08610
    [233]	train-logloss:0.00613	eval-logloss:0.08611
    [234]	train-logloss:0.00612	eval-logloss:0.08617
    [235]	train-logloss:0.00611	eval-logloss:0.08626
    [236]	train-logloss:0.00610	eval-logloss:0.08629
    [237]	train-logloss:0.00609	eval-logloss:0.08622
    [238]	train-logloss:0.00608	eval-logloss:0.08639
    [239]	train-logloss:0.00607	eval-logloss:0.08634
    [240]	train-logloss:0.00606	eval-logloss:0.08618
    [241]	train-logloss:0.00605	eval-logloss:0.08619
    [242]	train-logloss:0.00604	eval-logloss:0.08625
    [243]	train-logloss:0.00602	eval-logloss:0.08626
    [244]	train-logloss:0.00601	eval-logloss:0.08629
    [245]	train-logloss:0.00600	eval-logloss:0.08622
    [246]	train-logloss:0.00599	eval-logloss:0.08640
    [247]	train-logloss:0.00598	eval-logloss:0.08635
    [248]	train-logloss:0.00597	eval-logloss:0.08628
    [249]	train-logloss:0.00596	eval-logloss:0.08645
    [250]	train-logloss:0.00595	eval-logloss:0.08629
    [251]	train-logloss:0.00594	eval-logloss:0.08631
    [252]	train-logloss:0.00593	eval-logloss:0.08636
    [253]	train-logloss:0.00592	eval-logloss:0.08639
    [254]	train-logloss:0.00591	eval-logloss:0.08649
    [255]	train-logloss:0.00590	eval-logloss:0.08644
    [256]	train-logloss:0.00589	eval-logloss:0.08629
    [257]	train-logloss:0.00588	eval-logloss:0.08646
    [258]	train-logloss:0.00587	eval-logloss:0.08639
    [259]	train-logloss:0.00586	eval-logloss:0.08644
    [260]	train-logloss:0.00585	eval-logloss:0.08646
    [261]	train-logloss:0.00585	eval-logloss:0.08649
    [262]	train-logloss:0.00584	eval-logloss:0.08644
    [263]	train-logloss:0.00583	eval-logloss:0.08647
    [264]	train-logloss:0.00582	eval-logloss:0.08632
    [265]	train-logloss:0.00581	eval-logloss:0.08649
    [266]	train-logloss:0.00580	eval-logloss:0.08654
    [267]	train-logloss:0.00579	eval-logloss:0.08647
    [268]	train-logloss:0.00578	eval-logloss:0.08650
    [269]	train-logloss:0.00577	eval-logloss:0.08652
    [270]	train-logloss:0.00576	eval-logloss:0.08669
    [271]	train-logloss:0.00575	eval-logloss:0.08674
    [272]	train-logloss:0.00575	eval-logloss:0.08683
    [273]	train-logloss:0.00574	eval-logloss:0.08668
    [274]	train-logloss:0.00573	eval-logloss:0.08664
    [275]	train-logloss:0.00572	eval-logloss:0.08650
    [276]	train-logloss:0.00571	eval-logloss:0.08636
    [277]	train-logloss:0.00570	eval-logloss:0.08652
    [278]	train-logloss:0.00570	eval-logloss:0.08657
    [279]	train-logloss:0.00569	eval-logloss:0.08659
    [280]	train-logloss:0.00568	eval-logloss:0.08668
    [281]	train-logloss:0.00567	eval-logloss:0.08664
    [282]	train-logloss:0.00566	eval-logloss:0.08650
    [283]	train-logloss:0.00566	eval-logloss:0.08636
    [284]	train-logloss:0.00565	eval-logloss:0.08640
    [285]	train-logloss:0.00564	eval-logloss:0.08643
    [286]	train-logloss:0.00563	eval-logloss:0.08646
    [287]	train-logloss:0.00562	eval-logloss:0.08650
    [288]	train-logloss:0.00562	eval-logloss:0.08637
    [289]	train-logloss:0.00561	eval-logloss:0.08646
    [290]	train-logloss:0.00560	eval-logloss:0.08645
    [291]	train-logloss:0.00559	eval-logloss:0.08632
    [292]	train-logloss:0.00558	eval-logloss:0.08628
    [293]	train-logloss:0.00558	eval-logloss:0.08615
    [294]	train-logloss:0.00557	eval-logloss:0.08620
    [295]	train-logloss:0.00556	eval-logloss:0.08622
    [296]	train-logloss:0.00556	eval-logloss:0.08631
    [297]	train-logloss:0.00555	eval-logloss:0.08618
    [298]	train-logloss:0.00554	eval-logloss:0.08626
    [299]	train-logloss:0.00553	eval-logloss:0.08613
    [300]	train-logloss:0.00553	eval-logloss:0.08618
    [301]	train-logloss:0.00552	eval-logloss:0.08605
    [302]	train-logloss:0.00551	eval-logloss:0.08602
    [303]	train-logloss:0.00551	eval-logloss:0.08610
    [304]	train-logloss:0.00550	eval-logloss:0.08598
    [305]	train-logloss:0.00549	eval-logloss:0.08606
    [306]	train-logloss:0.00548	eval-logloss:0.08597
    [307]	train-logloss:0.00548	eval-logloss:0.08600
    [308]	train-logloss:0.00547	eval-logloss:0.08600
    [309]	train-logloss:0.00546	eval-logloss:0.08588
    [310]	train-logloss:0.00546	eval-logloss:0.08592
    [311]	train-logloss:0.00545	eval-logloss:0.08595
    

train으로 학습 수행하면 반복 시 train-error와 eval-logloss가 지속적으로 감소

* * *

xgboost 이용해 모델 학습이 완료됐으면 이를 이용해 테스트 데이터셋에 예측 수행

파이썬 래퍼 XGBoost는 train()함수를 호출해 학습이 완료된 모델 객체 반환, 이 모델 객체는 예측을 위해 predict()메서드 이용

주의 : 사이킷런 predict()메서드는 예측 결괏값(즉 0,1) 반환, xgboost의 predict()는 예측 결괏값이 아닌 예측 결과를 추정할 수 있는 확률 값을 반환


```python
pred_probs = xgb_model.predict(dtest)
print('predict() 수행 결괏값을 10개만 표시, 예측 확룰값으로 표시됨')
print(np.round(pred_probs[:10], 3))

# 예측 확률이 0.5보다 크면 1, 그렇지 않으면 0으로 예측값 결정해 리스트 객체인 preds에 저장
preds = [ 1 if x > 0.5 else 0 for x in pred_probs ]
print('예측값 10개만 표시:', preds[:10])
```

    predict() 수행 결괏값을 10개만 표시, 예측 확룰값으로 표시됨
    [0.934 0.003 0.91  0.094 0.993 1.    1.    0.999 0.997 0.   ]
    예측값 10개만 표시: [1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    


```python
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score

def get_clf_eval(y_test , pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    roc_auc = roc_auc_score(y_test, pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
```


```python
get_clf_eval(y_test, preds)
```

    오차 행렬
    [[35  2]
     [ 1 76]]
    정확도: 0.9737, 정밀도: 0.9744, 재현율: 0.9870,    F1: 0.9806, AUC:0.9665
    

### xgboost 패키지 내에 내장된 시각화 기능 수행

* plot_importance() : 피처의 중요도를 막대그래프 형식으로 나타냄. 호출 시 파라미터로 앞에서 학습이 완료된 모델 객체 및 맷플롯립의 ax객체를 입력

    -> 유의할 점: xgboost 넘파이 기반의 피처 데이터로 학습 시에 피처명을 제대로 알 수 없음
    
    -> f0, f1과 같이 피처 순서별로 f자 뒤에 순서를 붙여서 X축에 피처들로 나열


```python
from xgboost import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)
```




    <AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_18_1.png)
    


* to_graphviz() : 트리 기반 규칙 구조 그릴 수 있음 

    -> xgboost.to_graphviz()내에 파라미터로 학습이 완료된 모델 객체와 Graphviz가 참조할 파일명 입력


```python
xgboost.to_graphviz(xgb_model)
```




    
![svg](output_20_0.svg)
    



파이썬 래퍼 XGBoost는 사이킷런의 GridSearchCV 와 유사하게 데이터셋에 대한 교차 검증 수행 후 최적 파라미터를 구할 수 있는 방법을 cv() API로 제공

xgb.cv의 반환값은 DataFrame 형태

* * *

## 사이킷런 래퍼 XGBoost의 개요 및 적용

* XGBClassifier : 분류를 위한 래퍼 클래스
* XGBRegressor : 회귀를 위한 래퍼 클래스

> 기존 xgboost 모듈의 네이티브 하이퍼 파라미터 변경 사항
> * eta -> learning_rate
> * sub_sample -> subsample
> * lambda -> reg_lambda
> * alpha -> reg_alpha


```python
# 사이킷런 래퍼 XGBoost 클래스인 XGBClassifier 임포트
from xgboost import XGBClassifier

evals = [(X_test, y_test)]
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(X_train , y_train,  early_stopping_rounds=400,eval_set=evals, eval_metric="logloss",  verbose=True)
w_preds = xgb_wrapper.predict(X_test)
```

    [0]	validation_0-logloss:0.61352
    [1]	validation_0-logloss:0.54784
    [2]	validation_0-logloss:0.49425
    [3]	validation_0-logloss:0.44799
    [4]	validation_0-logloss:0.40911
    [5]	validation_0-logloss:0.37498
    [6]	validation_0-logloss:0.34571
    [7]	validation_0-logloss:0.32053
    [8]	validation_0-logloss:0.29721
    [9]	validation_0-logloss:0.27799
    [10]	validation_0-logloss:0.26030
    [11]	validation_0-logloss:0.24604
    [12]	validation_0-logloss:0.23156
    [13]	validation_0-logloss:0.22005
    [14]	validation_0-logloss:0.20857
    [15]	validation_0-logloss:0.19999
    [16]	validation_0-logloss:0.19012
    [17]	validation_0-logloss:0.18182
    [18]	validation_0-logloss:0.17473
    [19]	validation_0-logloss:0.16766
    [20]	validation_0-logloss:0.15820
    [21]	validation_0-logloss:0.15473
    [22]	validation_0-logloss:0.14895
    [23]	validation_0-logloss:0.14331
    [24]	validation_0-logloss:0.13634
    [25]	validation_0-logloss:0.13278
    [26]	validation_0-logloss:0.12791
    [27]	validation_0-logloss:0.12526
    [28]	validation_0-logloss:0.11998
    [29]	validation_0-logloss:0.11641
    [30]	validation_0-logloss:0.11450
    [31]	validation_0-logloss:0.11257
    [32]	validation_0-logloss:0.11154
    [33]	validation_0-logloss:0.10868
    [34]	validation_0-logloss:0.10668
    [35]	validation_0-logloss:0.10421
    [36]	validation_0-logloss:0.10296
    [37]	validation_0-logloss:0.10058
    [38]	validation_0-logloss:0.09868
    [39]	validation_0-logloss:0.09644
    [40]	validation_0-logloss:0.09587
    [41]	validation_0-logloss:0.09424
    [42]	validation_0-logloss:0.09471
    [43]	validation_0-logloss:0.09427
    [44]	validation_0-logloss:0.09389
    [45]	validation_0-logloss:0.09418
    [46]	validation_0-logloss:0.09402
    [47]	validation_0-logloss:0.09236
    [48]	validation_0-logloss:0.09301
    [49]	validation_0-logloss:0.09127
    [50]	validation_0-logloss:0.09005
    [51]	validation_0-logloss:0.08961
    [52]	validation_0-logloss:0.08958
    [53]	validation_0-logloss:0.09070
    [54]	validation_0-logloss:0.08958
    [55]	validation_0-logloss:0.09036
    [56]	validation_0-logloss:0.09159
    [57]	validation_0-logloss:0.09153
    [58]	validation_0-logloss:0.09199
    [59]	validation_0-logloss:0.09195
    [60]	validation_0-logloss:0.09194
    [61]	validation_0-logloss:0.09146
    [62]	validation_0-logloss:0.09031
    [63]	validation_0-logloss:0.08941
    [64]	validation_0-logloss:0.08972
    [65]	validation_0-logloss:0.08974
    [66]	validation_0-logloss:0.08962
    [67]	validation_0-logloss:0.08873
    [68]	validation_0-logloss:0.08862
    [69]	validation_0-logloss:0.08974
    [70]	validation_0-logloss:0.08998
    [71]	validation_0-logloss:0.08978
    [72]	validation_0-logloss:0.08958
    [73]	validation_0-logloss:0.08953
    [74]	validation_0-logloss:0.08875
    [75]	validation_0-logloss:0.08860
    [76]	validation_0-logloss:0.08812
    [77]	validation_0-logloss:0.08840
    [78]	validation_0-logloss:0.08874
    [79]	validation_0-logloss:0.08815
    [80]	validation_0-logloss:0.08758
    [81]	validation_0-logloss:0.08741
    [82]	validation_0-logloss:0.08849
    [83]	validation_0-logloss:0.08858
    [84]	validation_0-logloss:0.08807
    [85]	validation_0-logloss:0.08764
    [86]	validation_0-logloss:0.08742
    [87]	validation_0-logloss:0.08761
    [88]	validation_0-logloss:0.08707
    [89]	validation_0-logloss:0.08727
    [90]	validation_0-logloss:0.08716
    [91]	validation_0-logloss:0.08696
    [92]	validation_0-logloss:0.08717
    [93]	validation_0-logloss:0.08707
    [94]	validation_0-logloss:0.08659
    [95]	validation_0-logloss:0.08612
    [96]	validation_0-logloss:0.08714
    [97]	validation_0-logloss:0.08677
    [98]	validation_0-logloss:0.08669
    [99]	validation_0-logloss:0.08655
    [100]	validation_0-logloss:0.08650
    [101]	validation_0-logloss:0.08641
    [102]	validation_0-logloss:0.08629
    [103]	validation_0-logloss:0.08626
    [104]	validation_0-logloss:0.08683
    [105]	validation_0-logloss:0.08677
    [106]	validation_0-logloss:0.08732
    [107]	validation_0-logloss:0.08730
    [108]	validation_0-logloss:0.08728
    [109]	validation_0-logloss:0.08730
    [110]	validation_0-logloss:0.08729
    [111]	validation_0-logloss:0.08800
    [112]	validation_0-logloss:0.08794
    [113]	validation_0-logloss:0.08784
    [114]	validation_0-logloss:0.08807
    [115]	validation_0-logloss:0.08765
    [116]	validation_0-logloss:0.08730
    [117]	validation_0-logloss:0.08780
    [118]	validation_0-logloss:0.08775
    [119]	validation_0-logloss:0.08768
    [120]	validation_0-logloss:0.08763
    [121]	validation_0-logloss:0.08757
    [122]	validation_0-logloss:0.08755
    [123]	validation_0-logloss:0.08716
    [124]	validation_0-logloss:0.08767
    [125]	validation_0-logloss:0.08774
    [126]	validation_0-logloss:0.08828
    [127]	validation_0-logloss:0.08831
    [128]	validation_0-logloss:0.08827
    [129]	validation_0-logloss:0.08789
    [130]	validation_0-logloss:0.08886
    [131]	validation_0-logloss:0.08868
    [132]	validation_0-logloss:0.08874
    [133]	validation_0-logloss:0.08922
    [134]	validation_0-logloss:0.08918
    [135]	validation_0-logloss:0.08882
    [136]	validation_0-logloss:0.08851
    [137]	validation_0-logloss:0.08848
    [138]	validation_0-logloss:0.08839
    [139]	validation_0-logloss:0.08915
    [140]	validation_0-logloss:0.08911
    [141]	validation_0-logloss:0.08876
    [142]	validation_0-logloss:0.08868
    [143]	validation_0-logloss:0.08839
    [144]	validation_0-logloss:0.08927
    [145]	validation_0-logloss:0.08924
    [146]	validation_0-logloss:0.08914
    [147]	validation_0-logloss:0.08891
    [148]	validation_0-logloss:0.08942
    [149]	validation_0-logloss:0.08939
    [150]	validation_0-logloss:0.08911
    [151]	validation_0-logloss:0.08873
    [152]	validation_0-logloss:0.08872
    [153]	validation_0-logloss:0.08848
    [154]	validation_0-logloss:0.08847
    [155]	validation_0-logloss:0.08854
    [156]	validation_0-logloss:0.08852
    [157]	validation_0-logloss:0.08855
    [158]	validation_0-logloss:0.08828
    [159]	validation_0-logloss:0.08830
    [160]	validation_0-logloss:0.08828
    [161]	validation_0-logloss:0.08801
    [162]	validation_0-logloss:0.08776
    [163]	validation_0-logloss:0.08778
    [164]	validation_0-logloss:0.08778
    [165]	validation_0-logloss:0.08752
    [166]	validation_0-logloss:0.08754
    [167]	validation_0-logloss:0.08764
    [168]	validation_0-logloss:0.08739
    [169]	validation_0-logloss:0.08738
    [170]	validation_0-logloss:0.08730
    [171]	validation_0-logloss:0.08737
    [172]	validation_0-logloss:0.08740
    [173]	validation_0-logloss:0.08739
    [174]	validation_0-logloss:0.08713
    [175]	validation_0-logloss:0.08716
    [176]	validation_0-logloss:0.08696
    [177]	validation_0-logloss:0.08705
    [178]	validation_0-logloss:0.08697
    [179]	validation_0-logloss:0.08697
    [180]	validation_0-logloss:0.08704
    [181]	validation_0-logloss:0.08680
    [182]	validation_0-logloss:0.08683
    [183]	validation_0-logloss:0.08658
    [184]	validation_0-logloss:0.08659
    [185]	validation_0-logloss:0.08661
    [186]	validation_0-logloss:0.08637
    [187]	validation_0-logloss:0.08637
    [188]	validation_0-logloss:0.08630
    [189]	validation_0-logloss:0.08610
    [190]	validation_0-logloss:0.08602
    [191]	validation_0-logloss:0.08605
    [192]	validation_0-logloss:0.08615
    [193]	validation_0-logloss:0.08592
    [194]	validation_0-logloss:0.08592
    [195]	validation_0-logloss:0.08598
    [196]	validation_0-logloss:0.08601
    [197]	validation_0-logloss:0.08592
    [198]	validation_0-logloss:0.08585
    [199]	validation_0-logloss:0.08587
    [200]	validation_0-logloss:0.08589
    [201]	validation_0-logloss:0.08595
    [202]	validation_0-logloss:0.08573
    [203]	validation_0-logloss:0.08573
    [204]	validation_0-logloss:0.08575
    [205]	validation_0-logloss:0.08582
    [206]	validation_0-logloss:0.08584
    [207]	validation_0-logloss:0.08578
    [208]	validation_0-logloss:0.08569
    [209]	validation_0-logloss:0.08571
    [210]	validation_0-logloss:0.08581
    [211]	validation_0-logloss:0.08559
    [212]	validation_0-logloss:0.08580
    [213]	validation_0-logloss:0.08581
    [214]	validation_0-logloss:0.08574
    [215]	validation_0-logloss:0.08566
    [216]	validation_0-logloss:0.08584
    [217]	validation_0-logloss:0.08563
    [218]	validation_0-logloss:0.08573
    [219]	validation_0-logloss:0.08578
    [220]	validation_0-logloss:0.08579
    [221]	validation_0-logloss:0.08582
    [222]	validation_0-logloss:0.08576
    [223]	validation_0-logloss:0.08567
    [224]	validation_0-logloss:0.08586
    [225]	validation_0-logloss:0.08587
    [226]	validation_0-logloss:0.08593
    [227]	validation_0-logloss:0.08595
    [228]	validation_0-logloss:0.08587
    [229]	validation_0-logloss:0.08606
    [230]	validation_0-logloss:0.08600
    [231]	validation_0-logloss:0.08592
    [232]	validation_0-logloss:0.08610
    [233]	validation_0-logloss:0.08611
    [234]	validation_0-logloss:0.08617
    [235]	validation_0-logloss:0.08626
    [236]	validation_0-logloss:0.08629
    [237]	validation_0-logloss:0.08622
    [238]	validation_0-logloss:0.08639
    [239]	validation_0-logloss:0.08634
    [240]	validation_0-logloss:0.08618
    [241]	validation_0-logloss:0.08619
    [242]	validation_0-logloss:0.08625
    [243]	validation_0-logloss:0.08626
    [244]	validation_0-logloss:0.08629
    [245]	validation_0-logloss:0.08622
    [246]	validation_0-logloss:0.08640
    [247]	validation_0-logloss:0.08635
    [248]	validation_0-logloss:0.08628
    [249]	validation_0-logloss:0.08645
    [250]	validation_0-logloss:0.08629
    [251]	validation_0-logloss:0.08631
    [252]	validation_0-logloss:0.08636
    [253]	validation_0-logloss:0.08639
    [254]	validation_0-logloss:0.08649
    [255]	validation_0-logloss:0.08644
    [256]	validation_0-logloss:0.08629
    [257]	validation_0-logloss:0.08646
    [258]	validation_0-logloss:0.08639
    [259]	validation_0-logloss:0.08644
    [260]	validation_0-logloss:0.08646
    [261]	validation_0-logloss:0.08649
    [262]	validation_0-logloss:0.08644
    [263]	validation_0-logloss:0.08647
    [264]	validation_0-logloss:0.08632
    [265]	validation_0-logloss:0.08649
    [266]	validation_0-logloss:0.08654
    [267]	validation_0-logloss:0.08647
    [268]	validation_0-logloss:0.08650
    [269]	validation_0-logloss:0.08652
    [270]	validation_0-logloss:0.08669
    [271]	validation_0-logloss:0.08674
    [272]	validation_0-logloss:0.08683
    [273]	validation_0-logloss:0.08668
    [274]	validation_0-logloss:0.08664
    [275]	validation_0-logloss:0.08650
    [276]	validation_0-logloss:0.08636
    [277]	validation_0-logloss:0.08652
    [278]	validation_0-logloss:0.08657
    [279]	validation_0-logloss:0.08659
    [280]	validation_0-logloss:0.08668
    [281]	validation_0-logloss:0.08664
    [282]	validation_0-logloss:0.08650
    [283]	validation_0-logloss:0.08636
    [284]	validation_0-logloss:0.08640
    [285]	validation_0-logloss:0.08643
    [286]	validation_0-logloss:0.08646
    [287]	validation_0-logloss:0.08650
    [288]	validation_0-logloss:0.08637
    [289]	validation_0-logloss:0.08646
    [290]	validation_0-logloss:0.08645
    [291]	validation_0-logloss:0.08632
    [292]	validation_0-logloss:0.08628
    [293]	validation_0-logloss:0.08615
    [294]	validation_0-logloss:0.08620
    [295]	validation_0-logloss:0.08622
    [296]	validation_0-logloss:0.08631
    [297]	validation_0-logloss:0.08618
    [298]	validation_0-logloss:0.08626
    [299]	validation_0-logloss:0.08613
    [300]	validation_0-logloss:0.08618
    [301]	validation_0-logloss:0.08605
    [302]	validation_0-logloss:0.08602
    [303]	validation_0-logloss:0.08610
    [304]	validation_0-logloss:0.08598
    [305]	validation_0-logloss:0.08606
    [306]	validation_0-logloss:0.08597
    [307]	validation_0-logloss:0.08600
    [308]	validation_0-logloss:0.08600
    [309]	validation_0-logloss:0.08588
    [310]	validation_0-logloss:0.08592
    [311]	validation_0-logloss:0.08595
    [312]	validation_0-logloss:0.08603
    [313]	validation_0-logloss:0.08611
    [314]	validation_0-logloss:0.08599
    [315]	validation_0-logloss:0.08590
    [316]	validation_0-logloss:0.08595
    [317]	validation_0-logloss:0.08598
    [318]	validation_0-logloss:0.08600
    [319]	validation_0-logloss:0.08588
    [320]	validation_0-logloss:0.08597
    [321]	validation_0-logloss:0.08605
    [322]	validation_0-logloss:0.08609
    [323]	validation_0-logloss:0.08598
    [324]	validation_0-logloss:0.08598
    [325]	validation_0-logloss:0.08590
    [326]	validation_0-logloss:0.08578
    [327]	validation_0-logloss:0.08586
    [328]	validation_0-logloss:0.08594
    [329]	validation_0-logloss:0.08582
    [330]	validation_0-logloss:0.08587
    [331]	validation_0-logloss:0.08589
    [332]	validation_0-logloss:0.08592
    [333]	validation_0-logloss:0.08584
    [334]	validation_0-logloss:0.08574
    [335]	validation_0-logloss:0.08582
    [336]	validation_0-logloss:0.08589
    [337]	validation_0-logloss:0.08594
    [338]	validation_0-logloss:0.08583
    [339]	validation_0-logloss:0.08591
    [340]	validation_0-logloss:0.08583
    [341]	validation_0-logloss:0.08573
    [342]	validation_0-logloss:0.08568
    [343]	validation_0-logloss:0.08572
    [344]	validation_0-logloss:0.08580
    [345]	validation_0-logloss:0.08582
    [346]	validation_0-logloss:0.08571
    [347]	validation_0-logloss:0.08579
    [348]	validation_0-logloss:0.08583
    [349]	validation_0-logloss:0.08573
    [350]	validation_0-logloss:0.08566
    [351]	validation_0-logloss:0.08573
    [352]	validation_0-logloss:0.08581
    [353]	validation_0-logloss:0.08571
    [354]	validation_0-logloss:0.08566
    [355]	validation_0-logloss:0.08570
    [356]	validation_0-logloss:0.08563
    [357]	validation_0-logloss:0.08553
    [358]	validation_0-logloss:0.08560
    [359]	validation_0-logloss:0.08568
    [360]	validation_0-logloss:0.08558
    [361]	validation_0-logloss:0.08560
    [362]	validation_0-logloss:0.08564
    [363]	validation_0-logloss:0.08571
    [364]	validation_0-logloss:0.08579
    [365]	validation_0-logloss:0.08569
    [366]	validation_0-logloss:0.08573
    [367]	validation_0-logloss:0.08568
    [368]	validation_0-logloss:0.08559
    [369]	validation_0-logloss:0.08552
    [370]	validation_0-logloss:0.08559
    [371]	validation_0-logloss:0.08550
    [372]	validation_0-logloss:0.08556
    [373]	validation_0-logloss:0.08561
    [374]	validation_0-logloss:0.08563
    [375]	validation_0-logloss:0.08553
    [376]	validation_0-logloss:0.08561
    [377]	validation_0-logloss:0.08567
    [378]	validation_0-logloss:0.08571
    [379]	validation_0-logloss:0.08562
    [380]	validation_0-logloss:0.08558
    [381]	validation_0-logloss:0.08562
    [382]	validation_0-logloss:0.08564
    [383]	validation_0-logloss:0.08555
    [384]	validation_0-logloss:0.08562
    [385]	validation_0-logloss:0.08562
    [386]	validation_0-logloss:0.08555
    [387]	validation_0-logloss:0.08546
    [388]	validation_0-logloss:0.08550
    [389]	validation_0-logloss:0.08546
    [390]	validation_0-logloss:0.08532
    [391]	validation_0-logloss:0.08539
    [392]	validation_0-logloss:0.08530
    [393]	validation_0-logloss:0.08537
    [394]	validation_0-logloss:0.08530
    [395]	validation_0-logloss:0.08537
    [396]	validation_0-logloss:0.08528
    [397]	validation_0-logloss:0.08532
    [398]	validation_0-logloss:0.08528
    [399]	validation_0-logloss:0.08520
    


```python
get_clf_eval(y_test , w_preds)
```

    오차 행렬
    [[35  2]
     [ 1 76]]
    정확도: 0.9737, 정밀도: 0.9744, 재현율: 0.9870,    F1: 0.9806, AUC:0.9665
    


```python
from xgboost import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(10, 12))
# 사이킷런 wrapper 클래스 입력해도 무방
plot_importance(xgb_wrapper, ax=ax)
```




    <AxesSubplot:title={'center':'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_24_1.png)
    

