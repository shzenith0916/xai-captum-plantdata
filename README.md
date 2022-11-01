# xai-captum-plantdata

인턴십 및 졸업논문 프로젝트로, 식물 및 씨앗 이미지 데이터세트에 대해서 회사에서 사용하는 모델이,  이미지 데이터의 어떤 부분 혹은 패턴으로 모델이 그런 예측을 하는지에 대하여 공부 및 실험한 프로젝트
식물의 종자를 파는 Rijk Zwaan회사에서 제공한 이미지 데이터와 사용하는 회귀모델을 attribution 모델들에 적용하여 테스트 진행.
였고, 실제 눈으로 확인가능한 결과값을 보여주었습니다. 

### 프로젝트 사용 라이브러리
PyTorch Grad-CAM(Gradient-weighted Class Activation Mapping) implementation code [by kazuto](https://github.com/kazuto1011/grad-cam-pytorch) 이용,
[PyTorch Captum Library](https://github.com/pytorch/captum), [Torchray Library](https://facebookresearch.github.io/TorchRay/attribution.html) Attribution 모델들 적용. 

### 이미지 데이터
(1) Cucumber Leaf 오이 잎 - Controlled environment taken image, Mobile phone taken image in  greenhouse <br>
(2) Pepper Seed 후추씨 - X-ray taken image <br>
(3) Asparagus 아스파라거스 - Mobile phone taken image <br>

### 전처리 
(1.1) 블랙박스 오이잎 이미지 - 특별히 제작한 까만 박스에서 촬영하여, 오이잎을 제외한 부분이 이미 까만색으로 전처리 없이 attribution 모델들 테스트 진행 <br>
(1.2) 그린하우스 오이잎 이미지 - 핸드폰으로 촬영된 이미지로 노이즈가 심해서, attribution 모델들 테스트 시 회사의 분류모델이 잘 예측하지 못하는 결과 확인. 다시 전처리 후, 
attribution 모델들 적용시, 분류 모델이 잘 예측하는 결과 확인 <br>
(2) 엑스레이 후추씨 이미지 - attribution 모델 적용시 회귀모델이 잘 예측하는 결과를 보였으나, 전처리 효과 확인 위해 따로 opencv에서 Image Denoising을 이용하여 전처리 후 
다시 테스트 진행 <br>
(3) 아스파라거스 이미지 - 샘플이미지 수가 약 200-300개로 데이터가 너무 작아, 윤곽을 세그먼트하여(image segmentation) 따로 저장하는 전처리 후 attribution 모델 테스트 진행  <br>

### 이미지 별 회사에서 사용하는 모델 
(1 오이잎이 병이 들었다 들지 않았다를 예측하는 회귀모델  <br>
(2) 후추 씨앗이 발아하기에 적합하다 부적합하다 등의 다섯단계로 예측하는 회귀모델 - abnormal(이상있음), no prime(적절하지 않음), light prime(살짝 적절함), prime(적절), before germination(발아전)  <br>
(3.1) 다 자란 아스파라거스가 좋은 퀄리티의 아스파라거스인지 예측하는 회귀 모델  <br>
(3.2) 프로젝트에서 회귀모델에 Unet을 적용한뒤, 모델 메트릭 값이 예전 모델과 비교하였을 때 약 8%정도 향상  <br>

### 이미지 별 테스트 
(1) Captum 라이브러리에 있는 다양한 어트리뷰션 모델들로 실험한 결과, 분류모델이 오이잎의 어느 부분 때문에 병이 들었다고 예측하는지 테스트 진행. 
핸드폰으로 촬영한 그린하우스 이미지 데이터에 대해서는 오이잎 부분을 제외한 백그라운드에서 꽤 많은 노이즈를 모델에서 주목하는 부분으로 잡아내는 결과.
반대로 오이잎을 제외한 백그라운드 부분이 까만 이미지 데이터세트에 대해서 어트리뷰션 모델 결과값은 회사에서 예측하였던대로 오이잎의 병든 부분에 분류 모델이 
주목하고 있다는 결과.
대부분의 attribution 모델들의 결과는 병든 부분을 일부 혹은 대부분을 하이라이트 했으나, 그 중 Grad-CAM 모델이 비교적 가장 정확하게 잎의 병든 부분을 하이라이트. <br> 
(2) 후추 씨앗 이미지의 어떤 부분/패턴이 후추 씨앗이 발아하기에 적합하다 부적합하다 등의 다섯단계로 예측하는 모델에 영향을 끼치는지에 대해서 다양한 attribution 모델들을 적용. 
Grad-CAM 모델의 이미지 결과값이 눈으로 읽고 이해하기에 가장 쉽게 후추 씨앗위로 하이라이트 부분을 표시해주었습니다. 
attribution 모델 적용 후 하이라이트된 이미지 데이터세트를 회사의 씨앗을 연구하는 부서로 넘겨 추후 더 연구할 수 있도록 하였습니다. <br> 
(3) Unet이 적용된 회귀모델에 attribution 모델들 적용하여, 아스파라거스 이미지의 어떤 부분이 하이라이트되며, 회귀모델이 예측에 주목하는 부분인지 어트리뷰션 결과값 데이터세트를 아스파라거스를 전담하는 부서에 전달 
