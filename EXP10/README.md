# Read Me! 

* 이번 exp 폴더 안에는 주피터 노트북 파일 외에 다른 부분도 포함되어 있어, 이를 설명할 read me를 간략하게 작성했습니다. 

-------------------
# 1. gif 디렉토리
  * ![image](https://user-images.githubusercontent.com/96944114/154039157-912e067a-351f-45d8-ac3f-4a51be6349b4.png)
  * 이번 exp에서 학습했던 모델의 결과 이미지와 그래프 이미지를 올려둔 폴더 입니다. 


| case. | 시도한 방법, 변경점  |
| --- | --- |
| 1.  | epoch : 50, learning rate : 1e-4  |
| 2.  | epoch : 300, batch size : 256, learning rate : 2e-4 |
| 3.  | epoch : 300, batch size : 128, learning rate : 3e-4 | 
| 4.  | epoch : 300, batch size : 256, optimizer : learning rate : 2e-4, beta_1=0.5 |


----------------------------------------


# 2. try_train_models
  * 이번 exp에서 학습에 사용했던 주피터 노트북 파일을 모아둔 디렉토리 입니다. 
  * case 2, 3, 4 를 실험했던 주피터 노트북 파일이 들어있습니다. 내용은 별반 다를 것이 없으나, 
    위 표의 시도한 방법대로 모델 학습이 이루어졌습니다. 
  * 또한, 이 모델들은 이전 case 에서 학습되고 checkpoint로 저장된 모델을 가져와 학습하였습니다.  


-----------------------


# 3. [E-10]CIFAR-10_img_maker.ipynb
  * 이번 EXP의 핵심 주피터 노트북 파일 입니다. 
  * exp 일련의 과정이 포함되어있고, 개선을 위해 시도한 방법, 결과, 회고 등이 포함되어 있습니다. 
