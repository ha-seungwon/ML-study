# [ML] 사이킷런 설치 및 구현 (m1 macbook)

# 사이킷런

파이썬 머신러닝 라이브러리 중 가장 많이 사용되는 라이브러리

- 가장 파이썬스러운 API 제공
- 머신러닝을 위한 매우 다양한 알고리즘과 개발을 위한 편리한 프레임워크와 API 제공
- 오랜 기간 실전 환경에서 검증, 매우 많은 환경에서 사용됨

## 사이킷런 설치 (m1)

```bash
#책에서 요하는 버전
pip install scikit-learn==1.0.2
#향후 최신버전을 원할시
pip install -U scikit-learn
```

![스크린샷 2022-09-12 오후 2.34.22.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/618f23be-3f98-4df3-814c-333c3bad5a63/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-09-12_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_2.34.22.png)

설치완료 후, jupyter notebook으로 버전확인

## 머신러닝 만들어 보기 - 붓꽃 품종 예측(분류 Classification)

붓꽃 데이터 세트는 꽃잎의 길이와 너비, 꽃받침의 길이와 너비 특징(Feature)을 기반으로 꽃의 품종을 예측

### 분류(Classification): 대표적인 지도학습(Supervised Learning)의 한 방법

지도학습은 학습을 위한 다양한 피처와 분류 결정값인 Lable 데이터로 모델을 학습한 뒤, 별도의 테스트 데이터 세트에서 미지의 레이블을 예측 → 정답이 주어진 데이터를 먼저 학습 한 뒤 미지의 정답을 예측

- 학습을 위해 주어진 데이터 세트: 학습 데이터 세트
- ML Model 예측 성능을 평가하기 위해 별도로 주어진 데이터 세트: 테스트 데이터 세트

---

### 주피터 노트북 생성 및 모듈 임포트

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
```

- sklearn.datasets : 사이킷런에서 자체적으로 제공하는 데이터 세트 생성 모듈 모임
- sklearn.tree: 트리 기반 ML 알고리즘을 구현한 클래스의 모임
- sklearn.model_selection: 학습데이터와 검증데이터, 예특데이터로 데이터를 분리하거나 최적의 파라미터로 평가하기 위한 다양한 모듈의 모임
    - 하이퍼 파라미터: 머신러닝 알고리즘별 최적의 학습을 위해 직접 입력하는 파라미터를 통칭 → 이를 이용해 알고리즘 성능을 튜닝할 수 있음

붓꽃 데이터 세트 생성: load_iris()

ML알고리즘은 의사 결정 트리(Decision Tree) 알고리즘으로, 이를 구현한 DecisionTreeClassifier을 적용

### 데이터프레임으로 피처와 데이터 값을 확인

```python
import pandas as pd

iris = load_iris()

iris_data = iris.data

iris_label = iris.target
print('iris target값:', iris_label)

print('iris target명:', iris.target_names)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df.head(3)
```

![스크린샷 2022-09-14 오후 3.07.41.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c64cc678-f165-4e84-9079-186a30e52da8/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-09-14_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.07.41.png)

Feature

- sepal length
- sepal width
- petal length
- petal width

Label

- 0 setosa 품종
- 1 versicolor 품종
- 2 virginica 품종

### 예측 정확도 구하기

```python
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)
#DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)
#학습 수행
dt_clf.fit(X_train, y_train)
#학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행
pred = dt_clf.predict(X_test)
from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
```

1. 데이터 세트 분리: 데이터를 학습 데이터와 테스트 데이터로 분리
- train_test_split()

학습 데이터와 테스트 데이터를 test_size 파라미터 입력값의 비율로 쉽게 분할

ex) test_size=0.2 → 테스트 데이터 20%, 학습용 데이터 80%로 전체 데이터 분할

X_train → 학습용 피처 데이터 세트

X_test → 테스트용 피처 데이터 세트

Y_train → 학습용 레이블 데이터 세트

Y_test → 테스트용 레이블 데이터 세트

1. 모델 학습: 학습 데이터를 기반으로 ML 알고리즘을 적용해 모델을 학습
- DecisionTreeClassifier 객체 생성
1. 예측 수행: 학습된 ML 모델을 이용해 테스트 데이터의 분류(즉, 붓꽃 종류)를 예측
- predict() 메서드에 테스트용 피처 데이터 세트를 입력
1. 평가: 이렇게 예측된 결괏값과 테스트 데이터의 실제 결괏값을 비교해 ML 모델 성능을 평가
- accuracy_score() 정확도 측정

## 사이킷런의 기반 프레임워크

### Estimator 이해 및 fit(), predict()메서드

ML 모델 학습: fit()

학습된 모델의 예측: predict()

Estimator: 학습, 예측

- Classifier 분류
    - 분류 구현 클래스
        - DecisionTreeClassfier
        - RandomForestClassfier
        - GraadientBoostingClassfier
        - GaussianNB
        - SVC
- Regressor 회귀
    - 회귀 구현 클래스
        - LinearRegression
        - Ridge
        - Lasso
        - RandomForestRegressor
        - GradientBoostingRegressor

### 내장된 예제 데이터 세트

사이킷런에 내장된 데이터 세트는 일반적으로 딕셔너리 형태

키는 보통 data, target, target_name, feature_names, DESCR로 구셩

- data: 피처의 데이터 셋을 가리킴
- target: 분류 시 레이블 값, 회귀일 떄는 숫자 결괏값 데이터 셋
- target_names: 개별 레이블의 이름
- feature_names: 피처의 이름
- DESCR: 데이터 셋에 대한 설명과 각 피처의 설명을 나타냄

data, target→ 넘파이 배열 ndarray 타입

target_names, feature_names → 넘파이 배열 또는 파이썬 리스트(list)타입

DESCR → 스트링 타입

```python
iris_data = load_iris()
print(type(iris_data))
```

`<class 'sklearn.utils.Bunch'>`

Bunch class는 파이선 딕셔너리 자료형과 유사

```python
keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들:', keys)
```

`붓꽃 데이터 세트의 키들: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])`

데이터 키는 피처들의 데이터 값을 가리킴

[데이터세트.data](http://데이터세트.data) 또는 데이터세트[’data’] 를 이용하여 사용