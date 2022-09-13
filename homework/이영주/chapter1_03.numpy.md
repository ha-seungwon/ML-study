# 01. 파이썬 기반의 머신러닝과 생태계 이해

## 03. 넘파이

* * * 

### 넘파이 개요

넘파이의 기반 데이터 타입: ndarray

* array() : 다양한 인자를 입력받아서 ndarray로 변환

생성된 ndarray의 shape 변수는 크기(행과 열의 수)를 튜플 형태로 가지고 있음 → 차원 알 수 있음

```python
import numpy as np
array1 = np.array([1,2,3])
print('array1 type:', type(array1)) 
print('array1 array 형태:',array1.shape)

array2 = np.array([[1,2,3],
                  [2,3,4]])
print('array2 type:',type(array2))
print('array2 array 형태:',array2.shape)

array3 = np.array([[1,2,3]])
print('array3 type:',type(array3))
print('array3 array 형태:', array3.shape)
```

```
[결과]
array1 type: <class 'numpy.ndarray'>
array1 array 형태: (3,) # 1차원 array로 3개의 데이터 가짐
array2 type: <class 'numpy.ndarray'>
array2 array 형태: (2, 3) # 2차원 array로 2개의 행과 3개의 열을 가져 2*3=6개의 데이터를 가짐
array3 type: <class 'numpy.ndarray'>
array3 array 형태: (1, 3) # 2차원 array로 1개의 행과 3개의 열을 가짐
```

같은 데이터라도 차원이 달라서 오류가 생기는 경우가 빈번하니 차원에 대한 이해를 확실히 해야 함

* * *

### ndarray의 데이터 타입

ndarray 내의 데이터 타입은 같은 데이터 타입만 가능

type 속성으로 데이터 타입 확인 가능

```python
list1 = [1,2,3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)
```

```
[결과]
<class 'list'>
<class 'numpy.ndarray'>
[1 2 3] int32 
```

서로 다른 데이터 유형이 섞여 있는 리스트를 ndarray로 변경하면 데이터 크기가 더 큰 데이터 타입으로 일괄적으로 형 변환

```python
array_int = np.array([1,2,3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

array_int1 = array_float.astype('int32')
print(array_int1, array_int1.dtype)

array_float1 = np.array([1.1,2.1,3.1])
array_int2 = array_float1.astype('int32')
print(array_int2, array_int2.dtype)
```

```python
[결과]
[1. 2. 3.] float64 # float형으로 저장했기 때문에 숫자 끝에 .붙음
[1 2 3] int32 # 다시 int 형으로 변환
[1 2 3] int32 # int 형으로 저장했기 때문에 소수점 뒷자리 없어짐
```

* * *

### ndarray를 편리하게 생성하기 - arange, zeros, ones

특정 크기나 차원을 가진 ndarray를 연속값이나 0또는 1로 초기화해 쉽게 생성할 필요가 있는 경우 사용

주로 테스트용으로 데이터를 만들거나 대규모의 데이터를 일괄적으로 초기화해야 할 경우에 사용

* arange() : 파이썬 표준 함수인 range()와 유사한 기능, 0부터 함수 인자 값 -1까지의 값을 순차적으로 ndarray의 데이터값으로 변환

```python
sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)
```

```
[결과]
[0 1 2 3 4 5 6 7 8 9] # 0부터 stop값인 10에서 -1을 한 9까지의 연속 숫자 값으로 구성된 ndarray
int32 (10,)
```

* zeros() : 함수 인자로 튜플 형태의 shape 값을 입력하면 모든 값을 0으로 채운 해당 형태의 ndarray를 반환

* ones() : 함수 인자로 튜플 형태의 shape 값을 입력하면 모든 값을 1로 채운 해당 형태의 ndarray 반환

함수 인자로 dtype을 정해주지 않으면 default로 ndarray를 float64형 데이터로 채움

```python
zero_array = np.zeros((3,2), dtype='int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)

one_array = np.ones((3,2))
print(one_array)
print(one_array.dtype, one_array.shape)
```

```
[결과]
[[0 0]
 [0 0]
 [0 0]]
int32 (3, 2)
[[1. 1.]
 [1. 1.]
 [1. 1.]]
float64 (3, 2)
```

* * *

### ndarray의 차원과 크기를 변경하는 reshape()

* reshape() : ndarray를 특정 차원 및 크기로 변환

```python
array1 = np.arange(10)
print('array1:\n', array1)

array2 = array1.reshape(2,5)
print('array2:\n', array2)

array3 = array1.reshape(5,2)
print('array3:\n', array3)
```

```
[결과]
array1:
 [0 1 2 3 4 5 6 7 8 9]
array2:
 [[0 1 2 3 4]
 [5 6 7 8 9]]
array3:
 [[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
```

-1을 인자로 사용하면 원래 ndarray와 호환되는 새로운 형태로 변환

```python
array1 = np.arange(10)
print(array1)
array2 = array1.reshape(-1,5)
print('array2 shape:', array2.shape)
array3 = array1.reshape(5,-1)
print('array3 shape:', array3.shape)
```

```
[결과]
[0 1 2 3 4 5 6 7 8 9]
array2 shape: (2, 5)
array3 shape: (5, 2)
```

array1은 1차원 ndarray이므로 0~9까지의 데이터 가짐, array2는 로우 인자가 -1, 칼럼 인자가 5. 

→ array1과 호환될 수 있는 2차원 ndarray로 변환하되 고정된 5개의 칼럼에 맞는 로우를 자동으로 새롭게 생성해 변환하라는 의미

⇒ 10개의 1차원 데이터와 호환될 수 있는 고정된 5개의 칼럼에 맞는 로우 개수는 2이므로 2*5의 2차원 ndarray로 변환

reshape는 -1인자와 함께 쓰였을 때 원본 ndarray가 어떤 형태라도, 원하는 차원의 ndarray로 변환됨을 보장

→ 여러 개의 넘파이 ndarray를 결합할 때 형태를 통일하기 위해 유용하게 사용됨

* * *

### 넘파이의 ndarray의 데이터 세트 선택하기 - 인덱싱(Indexing)

1. 특정한 데이터만 추출
    
    : 원하는 위치의 인덱스 값을 지정하면 해당 위치의 데이터 반환
    
2. 슬라이싱(Slicing)
    
    : 연속된 인덱스상의 ndarray 추출. ‘:’기호 사용, 시작 인덱스에서 종료 인덱스 -1 위치에 있는 데이터의 ndarray 반환
    
3. 팬시 인덱싱(Fancy Indexing)
    
    : 일정한 인덱싱 집합을 리스트 또는 ndarray 형태로 지정해 해당 위치에 있는 데이터의 ndarray 반환 
    
4. 불린 인덱싱(Boolean Indexing)
    
    : 특정 조건에 해당하는지 여부인 T/F값 인덱싱 집합을 기반으로 True에 해당하는 인덱스 위치에 있는 데이터의 ndarray 반환
    

> 단일 값 추출

```python
# 1부터 9까지의 1차원 ndarray 생성
array1 = np.arange(start=1, stop=10)
print('array1:', array1)
# index는 0부터 시작하므로 array1[2]는 3번째 index 위치의 데이터값 의미
value = array1[2]
print('value:', value)
print(type(value))
```

```
[결과]
array1: [1 2 3 4 5 6 7 8 9]
value: 3
<class 'numpy.int32'>
```

인덱스 -1은 맨 뒤의 데이터값 의미, -2는 뒤에서 두번째의 데이터값 의미

```python
array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3,3)
print(array2d)

print('(row=0, col=0) index 가리키는 값:',array2d[0,0])
print('(row=0, col=1) index 가리키는 값:',array2d[0,1])
print('(row=1, col=0) index 가리키는 값:',array2d[1,0])
print('(row=2, col=2) index 가리키는 값:',array2d[2,2])
```

```
[결과]
[[1 2 3]
 [4 5 6]
 [7 8 9]]
(row=0, col=0) index 가리키는 값: 1
(row=0, col=1) index 가리키는 값: 2
(row=1, col=0) index 가리키는 값: 4
(row=2, col=2) index 가리키는 값: 9
```

슬라이싱

단일 데이터값 추출을 제외하고 위의 세가지 방법으로 추출된 데이터 세트는 모두 ndarray 타입

```python
array1 = np.arange(start=1, stop=10)
array3 = array1[0:3]
print(array3)
print(type(array3))
```

```python
[결과]
[1 2 3]
<class 'numpy.ndarray'>
```

‘:’ 기호 앞/뒤에 시작/종료 인덱스를 생략하면 자동으로 맨 처음/맨 마지막 인덱스로 간주

팬시 인덱싱

리스트나 ndarray로 인덱스 집합을 지정하면 해당 위치의 인덱스에 해당하는 ndarray 반환

```python
array1d = np.arange(start=1,stop=10)
array2d = array1d.reshape(3,3)

array3 = array2d[[0,1],2]
print('array2d[[0,1],2] => ', array3.tolist())

array4 = array2d[[0,1],0:2]
print('array2d[[0,1],0:2] => ', array4.tolist())

array5 = array2d[[0,1]]
print('array2d[[0,1]] => ', array5.tolist())
```

```python
[결과]
array2d[[0,1],2] =>  [3, 6]
array2d[[0,1],0:2] =>  [[1, 2], [4, 5]]
array2d[[0,1]] =>  [[1, 2, 3], [4, 5, 6]] # ((0,:),(1,:)) 인덱싱 적용
```

불린 인덱싱

조건 필터링과 검색을 동시에 할 수 있기 떄문에 매우 자주 사용되는 인덱싱 방식

```python
array1d = np.arange(start=1,stop=10)
array3 = array1d[array1d>5]
print('array1d > 5 불린 인덱싱 결과 값:',array3)
```

```python
[결과]
array1d > 5 불린 인덱싱 결과 값: [6 7 8 9]
```

```python
array1d>5
```

```python
[결과]
array([False, False, False, False, False,  True,  True,  True,  True])
```

* * *

### 행렬의 정렬 - sort()와 argsort()

넘파이의 행렬 정렬

1. 넘파이에서 sort()를 호출하는 방식 - np.sort()
    
    : 원 행렬은 그대로 유지한 채 원 행렬의 정렬된 행렬 반환
    
2. 행렬 자체에서 sort()를 호출하는 방식 - ndarray.sort()
    
    : 원 행렬 자체를 정렬한 형태로 반환, 반환 값은 None
    

```python
org_array = np.array([3,1,9,5])
print('원본 행렬:',org_array)

sort_array1 = np.sort(org_array)
print('np.sort() 호출 후 반환된 정렬 행렬:',sort_array1)
print('np.sort() 호출 후 반환된 원본 행렬:',org_array)

sort_array2 = org_array.sort()
print('org_array.sort() 호출 후 반환된 행렬:',sort_array2)
print('org_array.sort() 호출 후 원본 행렬:',org_array)
```

```
[결과]
원본 행렬: [3 1 9 5]
np.sort() 호출 후 반환된 정렬 행렬: [1 3 5 9]
np.sort() 호출 후 반환된 원본 행렬: [3 1 9 5]
org_array.sort() 호출 후 반환된 행렬: None
org_array.sort() 호출 후 원본 행렬: [1 3 5 9]
```

정렬된 행렬의 인덱스를 반환하기

np.argsort(): 원본 행렬이 정렬되었을 때 기존 원본 행렬의 원소에 대한 인덱스를 필요로 할 때 사용

정렬 행렬의 원본 행렬 인덱스를 ndarray형으로 반환

```python
org_array = np.array([3,1,9,5])
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 오름차순 정렬 시 원본 행렬의 인덱스:', sort_indices)

sort_indices_desc = np.argsort(org_array)[::-1]
print(type(sort_indices_desc))
print('행렬 내림차순 정렬 시 원본 행렬의 인덱스:', sort_indices)
```

```
[결과]
<class 'numpy.ndarray'>
행렬 오름차순 정렬 시 원본 행렬의 인덱스: [1 0 3 2]
<class 'numpy.ndarray'>
행렬 내림차순 정렬 시 원본 행렬의 인덱스: [2 3 0 1]
```

넘파이는 메타 데이터를 가질 수 없어 실제 값과 그 값이 뜻하느 메타 데이터를 별도의 ndarray로 각각 가져야 함
→ 넘파이에서 argsort()의 활용도가 매우 높음

```python
import numpy as np

name_array = np.array(['John','Mike','Sarah','Kate','Samuel'])
score_array = np.array([78,95,84,98,88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스:',sort_indices_asc)
print('성적 오름차순으로 name_array의 이름 출력:',name_array[sort_indices_asc])
```

```
[결과]
성적 오름차순 정렬 시 score_array의 인덱스: [0 2 4 1 3]
성적 오름차순으로 name_array의 이름 출력: ['John' 'Sarah' 'Samuel' 'Mike' 'Kate']
```

* * *

### 선형대수 연산 - 행렬 내적과 전치 행렬 구하기

행렬 내적(행렬 곱)

np.dot() :  두 행렬 A와 B의 내적

```python
A = np.array([[1,2,3],
             [4,5,6]])
B = np.array([[7,8],
             [9,10],
             [11,12]])
dot_product = np.dot(A,B)
print('행렬 내적 결과:\n', dot_product)
```

```
[결과]
행렬 내적 결과:
 [[ 58  64]
 [139 154]]
```

전치 행렬

np.transpose() : 전치 행렬(원 행렬에서 행과 열 위치를 교환한 원소로 구성한 행렬)

```python
A = np.array([[1,2],
              [3,4]])
transpose_mat = np.transpose(A)
print('A의 전치 행렬:\n', transpose_mat)
```

```
A의 전치 행렬:
 [[1 3]
 [2 4]]
```
