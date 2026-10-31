# E0 시작 전 JPEG 체크섬 오류 수정

날짜: 2026-10-31 (KST). 구현 커밋: `6c4387a`.

## 원인

E0 시작 전 검증에서 `val/images/WEB07271.jpg`의 SHA256 불일치가 발생했다.
전체 검사 결과 이전 공개 모델 Validation에서 보정본 JPEG 6개가 변경돼 있었다.
원본 파일의 해시는 최초 manifest와 일치했다. E0는 학습 epoch에 진입하지 못했으며
이 실패에 대해 재개할 checkpoint는 없다.

해당 원본 JPEG에는 정상 종료 마커 `FF D9` 뒤에 `MediaType=default` 등의 부가
바이트가 있었다. Ultralytics 8.4.0의 `verify_image_label`은 마지막 두 바이트가
종료 마커인지 검사한다. 이 조건을 만족하지 않으면 Pillow로 JPEG를 quality 100에
재저장한다. 실제 로그의 `corrupt JPEG restored and saved`와 변경된 해시·파일 크기가
이 동작과 일치했다. 기존 준비 코드의 Pillow decode 검사는 부가 바이트를 허용했기
때문에 이 차이를 놓쳤다.

예: WEB07271.jpg 원본은 160,879 bytes, 최초 SHA256은
`987b3ca5dc7f2eddbb834ee459bfbb104dafb7396a470867cbeb8868906a0c54`.
자동 재저장된 보정본은 358,571 bytes, SHA256은
`22f368ddcfecb6c9af2324cca0deb22f4686ec853cc387e9d4b00d285c2e347a`였다.

## 수정

- 원본과 기존 보정본은 보존하고 `data/processed/dfire-fold1-v2`를 새로 준비했다.
- 전체 데이터에서 해당 형식 JPEG 37개를 찾았다. 보정본에서 종료 마커 뒤 바이트만
  제거하고 재압축하지 않았다. 원본·보정본의 RGB 크기와 픽셀이 완전히 같은지 확인했다.
- 원본 `source_image_sha256`, 실제 입력 `image_sha256`, 제거한 바이트 수와 픽셀 보존
  여부를 manifest/audit에 각각 기록한다. 종료 마커가 없는 JPEG는 임의 재압축하지 않고 실패한다.
- 라벨 보정 정책, 이미지 수, 공식 분할과 객체 수는 이전과 같다.
- 학습 loader 준비 후와 학습·평가 종료 시에 입력 해시와 fingerprint를 다시 확인한다.
- 사전 검증 실패 시 빈 학습 run 디렉터리를 만들지 않는다.
- 이전 공개 모델 평가에 `invalidated.json`을 추가하고 비교·최종 선택·Test 진입에서 제외한다.
  과거 수치와 원래 실행 파일은 이력으로 보존한다.

새 데이터 fingerprint:
`c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`.

## 검증

단위 테스트 28개를 통과했다. 추가 회귀 테스트는 부가 바이트가 있는 JPEG를 생성하고
설치된 Ultralytics의 실제 `verify_image_label`을 두 번 호출해 재저장과 해시 변경이
발생하지 않음을 검증한다. 원본 보존, 픽셀 동일성, 실행 중 입력 변경 감지,
무효 평가의 선택 차단도 확인했다. 합성 fixture는 실제 모델 성능 실험이 아니다.

실제 데이터 재검증 및 E0 loader 점검의 상세 근거는
[003_jpeg_integrity_fix.json](003_jpeg_integrity_fix.json)에 기록한다.

- 실제 정규화 JPEG 37개에 설치된 Ultralytics verifier를 호출한 뒤 해시가 유지됐다.
- 원본 이미지 21,527개와 label 21,527개의 해시가 최초 원본 해시와 모두 일치했다.
- E0와 같은 640 / batch 16 / MuSGD 조건으로 Train 13,776 / Val 3,445 전체 loader를
  준비했다. Ultralytics scan 이후 전체 입력 해시와 fingerprint가 유지됐다.
- 첫 epoch 전에 의도적으로 종료한 loader 점검이며 optimizer update는 0회다.
  E0 50 epoch 성능 실험을 실행하거나 완료한 것으로 취급하지 않는다.

새 보정본에서 공개 모델 Validation 3,445장도 다시 실행해 완료했다.
실행 폴더는 `experiments/runs/val-published_20261031T100414Z_e58fb232/`이며,
종료 후 전체 입력 SHA256과 fingerprint 검사를 통과했다. 정확한 지표·설정·환경은
JSON 보고서의 `replacement_published_validation`에 보관한다.
공개 모델 학습 데이터와의 중복 여부는 여전히 미확인이므로 일반화 성능이나
게시자 Test 수치의 재현으로 해석하지 않는다. 최종 Test 성능은 평가하지 않았다.

## 사용자 실행

현재 설정은 새 보정본을 가리킨다. 다음 명령으로 E0를 처음부터 실행한다.

```powershell
.\.venv\Scripts\python.exe firelab.py train --id E0
```

이번 오류는 학습 전 발생했으므로 `resume` 명령을 사용하지 않는다.
