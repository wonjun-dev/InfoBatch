# Summary
과제 실험은 도커 이미지 pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel 기반에서 수행되었습니다.
도커 이미지 설치후 run_container.sh 을 통해 컨테이너를 실행 시킨 후에, 컨테이너 내에서 실험을 재현 할 수 있습니다.\
실험에 대한 결과는 리뷰 PDF에 첨부하였습니다.

# Preparing

### 컨테이너 실행

```sh
sh run_container.sh
```
혹시 volume mount의 경로가 맞지 않다면 위의 sh 파일의 docker 명령어의 -v 인자를 수정해주세요.

###  MLflow 설치
컨테이너 실행 후, 컨테이너 내부에서 mlflow를 설치해주세요.
```sh
pip install mlflow
```

# 파일 구성
모든 소스 코드는 src 폴더 안에 있습니다.\
실험을 수행하기 위한 메인 실행 파일은 res18_cifar_xx_xx.py 형태 입니다.\
Pruning을 위한 Policy 및 Dataset 코드는 src/utils 경로에 있습니다.\
학습에서 사용한 hyper-parameter는 config 경로에 있습니다.\
Pruning probability와 Annealing 값은 src/utils/policy.py의 PruningPolicy 인자를 통해 조절 할 수 있습니다.


**res18_cifarXX_whole.py**: 전체 데이터를 학습하는 실험 코드 입니다.\
**res18_cifarXX_ib.py**: InfoBatch를 통해 학습하는 실험 코드 입니다.\
**res18_cifarXX_ib_ma.py**: Moving average threshold를 사용하여 InfoBatch를 수행하는 실험 코드 입니다. (리뷰 pdf의 Exp-4)\
**res18_cifarXX_ib_rev.py** Pruning 되는 샘플을 학습하는 InfoBatch 실험 코드 입니다. (리뷰 pdf의 Exp-5)

# 실행
각 실험 코드 (res18_cifarXX_XX_XX.py) 를 실행하면 mlflow에 실험 로그들이 기록됩니다.
실험 로그는 mlflow ui를 통해서 확인 할 수 있습니다.
```sh
mlflow ui
```

# 수행한 실험
각 실험의 이름은 리뷰 PDF의 Experiment 섹션의 이름과 상응합니다.\
**Exp-1**: pruning 확률을 0.3, 0.5, 0.7 로 조정해가며 그에 따른 성능 및 학습 시간 변화\
**Exp-2**: Gradient scale이 적용되지 않았을 때의 성능 변화\
**Exp-3**: Annealing이 적용되지 않았을 때의 성능 변화\
**Exp-4**: Moving average threshold를 사용했을 떄의 성능 변화\
**Exp-5**: Pruning 되는 샘플을 학습 했을 때의 성능 변화