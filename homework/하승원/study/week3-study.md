# Week 3  - Study


# 1-4 Random Forest

앙상블 방식중 보팅의 한 모델로 결정 트리(Desicion Tree) 방식을 기반으로 한 하드 보팅 앙상블 모델이다. 

![다운로드](https://user-images.githubusercontent.com/74447373/190887326-6aa71a19-18e7-4c86-bfb1-39332e1124a2.jpeg)

# 1-5 GBM (Gradient Boosting Machine)

GBM 방식은 Boosting 방식을  Gradient descent 방식으로 최적화를 한 모델로 부스팅 모델중 한 방법이다.


##  Gradient descent
> 경사 하강법(傾斜下降法, Gradient descent)은 1차 근삿값 발견용 최적화 알고리즘이다. 기본 개념은 함수의 기울기(경사)를 구하고 경사의 반대 방향으로 계속 이동시켜 극값에 이를 때까지 반복시키는 것이다.

출처: https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95


이론적 의미는 위와 같다. 다음 값을 (기울기: Gradient)를 구할때 값이 최소가 되도록 x 값을 이동시켜 기울기를 찾아가는 방식 이해를 돕기 위해 밑의 그림을 첨부했다.

![gbm-gradient](https://user-images.githubusercontent.com/74447373/190888114-d6395e65-b449-45a7-9f70-5132183a413b.png)


## Gradient descent와 Boosting
Boosting방식은 여러 분류기를 순차적으로 실행하면서 발생하는 오류를 가중치를 두면서 다음 분류기에 가중치를 기반으로 예측을 하는 방식이다.
예측값 F(x)과 실제 값 y의 차이를 나타내는 손실 함수 (Loss Function)를 MSE Mean Squared Error를 2로 나눈 것으로 정의하면, 
이 손실 함수의 1차 미분값 (gradient)의 음수를 취한게 바로 잔차 (residual)가 된다.

손실 함수는 예측값과 실제값의 차이를 나타내는 함수이기 때문에 값이 작을수록 좋다.
따라서 이 손실 함수의 극소값을 찾는 데 gradient descent를 이용한 최적화 문제로 치환할 수 있다.
하지만 동시에 과적합이 될 수 있다는 단점이 있다.  다시말해 모델이 학습데이터에 맞추어 학습하다보면, 
다른 일반적인 데이터를 예측하는 데 어려움이 있을 수 있다. 이에 따라 GBM 기반 모델에는 정규화하는 알고리즘이 꼭 포함되어 있다.
예를 들어 learning_rate를 활용하여 정규화할 수 있는데, 예측값에 곱하여 예측값과 실제값과의 차이를 일부러 조금씩 남겨두는 것이다.


GBM 알고리즘은 tree를 단계적으로 하나씩(one-by-one) 결합해나가게 된다. 
즉 앞선 그림에서 보았던 것처럼 Sequential하게 각 tree의 예측 결과를 합산해나간다.



![다운로드](https://user-images.githubusercontent.com/74447373/190888932-2307af1c-f1ad-4c59-aae3-06aa7fda179b.png)

# 구동 방식

먼저 초기 값을 랜덤하게 지정을 해주고 이 값을 기준점으로 삼아서 부모 노드와 리프노드를 구성 그후 트리를 구성을 하여 패턴을 만들고 각 결과값과 비교를 한다.
이후 예측 결과가 실제 값과 다르다면 그 값을 에러라 판단하고 이후 분류기에서 가중치를 두고 다시 예측을 하는 방식이다.


![img](https://user-images.githubusercontent.com/74447373/190896631-d578d2d6-3613-4c69-8263-0c789b747c8e.png)




https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-15-Gradient-Boost?category=1057680
추가내용

# 1-6 XGBoost

XGBoost는 GBM 방식에서 과적합 문제를 방지하기 위한 기법이 추가된 방식이다. 

