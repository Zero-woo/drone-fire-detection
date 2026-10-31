# Colab inline backend 상속 오류 수정

날짜: 2026-11-01 (KST). 구현 커밋: `a4f2fead5507bd2b6dec59a33eec63575557ca63`.

## 관찰과 원인

사용자 로그는 Colab Python 3.13.16, PyTorch 2.10.0+cu128, Tesla T4 인식 후
`data.yaml error ... module://matplotlib_inline.backend_inline ... not a valid value for backend`로 중단됐다.
이 메시지에서 실제 예외는 Matplotlib의 backend 설정이다. Dataset 검사 함수가 해당 예외를
감싸 전달했으며, 로그의 dataset 경로 자체가 누락됐다는 의미는 아니다.

별도 가상환경은 노트북의 inline backend 패키지를 설치하지 않지만 부모 프로세스의
`MPLBACKEND` 환경 변수를 상속했다. 로컬의 고정 Matplotlib 3.10.8 환경에서도 이 값을
지정하고 `import matplotlib`을 실행하자 같은 ValueError와 종료 코드 1을 재현했다.

## 수정

- 노트북의 모든 subprocess에 `MPLBACKEND=Agg`와 저장소 `work/matplotlib` 설정 폴더를 전달한다.
- 부모 커널의 환경 사본을 사용하므로 커널의 inline 출력 설정은 변경하지 않는다.
- CLI, YOLO 초기화 및 비교 그래프 생성에서도 Matplotlib import 전에 같은 설정을 적용한다.
- 모델·데이터·학습률·epoch·증강 설정은 변경하지 않았다. 입력 ZIP 재생성도 필요하지 않다.
- checkpoint가 없는 failed run을 재개하려 할 때 노트북에서 명확한 안내를 제공한다.

## 검증

- 오류 환경을 상속한 새 프로세스에서 수정 경로를 실행하니 backend `Agg`와 종료 코드 0을 확인했다.
- PNG 파일 저장을 확인했다. 이는 그래프 렌더링 점검이며 모델 성능 실험이 아니다.
- 전체 단위 테스트 37개를 통과했다. subprocess 렌더링 회귀 테스트와, CUDA/Drive 환경을
  보존하면서 커널의 backend 환경 사본을 분리하는 노트북 테스트를 포함한다.
- 실제 Colab GPU에서 재실행하거나 학습 완료 여부를 확인하지는 않았다.

기계가 읽을 수 있는 근거는 [006_colab_backend_fix.json](006_colab_backend_fix.json)에 있다.
현재 열린 노트북에 적용할 셀은 [USAGE](../docs/USAGE.md)에 수록했다.

출처: [Matplotlib backend 선택](https://matplotlib.org/stable/users/explain/figure/backends.html),
[환경 변수 안내](https://matplotlib.org/stable/install/environment_variables_faq.html).
