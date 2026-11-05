# main 통합 전 검증 및 서버 실행 보존

관측 시각: 2026-11-05T16:55:28.903807+09:00.
이 문서는 회귀 테스트와 운영 상태의 검증 기록이다. 합성 fixture·probe는 실제 모델 실험이나 실제 Test 평가 결과가 아니다.
세부 실행 명령·로그 SHA256·소스/설정/테스트 SHA256·버전·Git 상태는 [동명 JSON](015_main_integration_verification.json)에 보존한다.

## 회귀 검증

- 서버 저장소의 `/root/drone-fire-detection/.venv/bin/python -m unittest discover -s tests -v`로 63개 테스트를 실행했고, 누락/실패 없이 모두 통과했다.
- unittest 기록 시간: 7.753초; 종료 코드 0. CPU만 사용하고 GPU를 추가 점유하지 않았다.
- 테스트 대상 소스 Git: `f1f802dd05903ec1265c550e67f19fa735ea720c`. 작업 트리 차이는 자동 생성된 진행 기록과 완료 E2 보고서였으며, 고정된 소스·설정 16개와 테스트 파일 5개의 SHA256을 로컬에서도 대조했다.
- 완료된 E1·E2-A·E2-C의 best/last 가중치, training/evaluation 연결, 데이터 fingerprint, 보고서 원본 SHA256을 서버에서 검증했다.
- 원격 main 기준에 있던 준비·구현·측정 커밋을 보존하며, E2 완료 결과와 이번 운영 검증은 별도 커밋으로 유지한다.

## 완료 측정과 진행 상태

- E2-A: 실제 60/60 epoch, patience=15, epoch_limit 종료. 공통 mAP50-95 0.45099. [측정 보고서](010_server_e2-a.md).
- E2-C: 실제 60/60 epoch, patience=15, epoch_limit 종료. 공통 mAP50-95 0.44756. [측정 보고서](010_server_e2-c.md).
- E1과 E2는 학습 예산이 다르므로 위 비교를 학습률 단독 효과로 해석하지 않는다.

| 실험 | 관측 상태 | 저장 epoch |
|---|---|---:|
| E1 | completed | 100 |
| E2-A | completed | 60 |
| E2-C | completed | 60 |
| E3-HSV | training | 13 |
| E3-Mosaic | pending | 0 |
| E4-800 | pending | 0 |

## 활성 서버 보존

- 서버 branch: `codex/shorter-server-experiments`; Git: `f1f802dd05903ec1265c550e67f19fa735ea720c`. 서버의 branch·checkout·소스·설정을 변경하지 않았다.
- 기존 관리자 PID 10187, 생성 시각 1793807768.03를 유지했고 생존을 확인했다.
- 현재 E3-HSV 작업자 PID 16596, 생성 시각 1793860060.96의 생존을 확인했다. stdin은 /dev/null, 출력은 기존 파일 로그이며 SIGHUP을 무시한다.
- 활성 계획 SHA256: `9e5b9a18b8ce808d58f222eaf99471af7683e965fd33e0e6dbd6586033e3396d`. 고정 소스·설정의 체크섬이 일치한다.
- main 병합 후에도 실행 중인 서버에서 pull이나 branch 전환을 하지 않는다. 현재 유한 큐를 기존 실행 조건으로 유지한다.
- E3 이후 실험은 완료로 표시하지 않는다. E6 조건·모델·threshold는 아직 동결하지 않았고 실제 Test 기록은 없다.
- 원본 사진·영상, 가중치, 전체 run 폴더, 환경, 인증정보는 병합 대상에 넣지 않는다.
