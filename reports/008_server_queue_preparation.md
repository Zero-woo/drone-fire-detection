# 서버 핵심 실험 큐 준비 및 기존 E0 재확인

2026-11-04 UTC. 인계 문서를 작업 지시로 적용했다. 이 기록은 기존 실측 결과의
재검증과 작업 관리자 준비이며 새로운 모델 실험 결과가 아니다.

- 백그라운드 exec 터미널의 유휴 중첩 Codex PID 4104만 종료했다.
  `firelab-codex` tmux와 현재 Codex PID 3761은 유지했다. 학습 작업자는 없었다.
- 기존 서버 E0 학습 및 Validation JSON의 `completed`, 50 epochs와 연결 경로를 확인했다.
  `best.pt` SHA256은 `793526814adbd0c2a9d31054c1c37563c3c77fdef6cded429f062b964a4ee85d`,
  `last.pt`는 `e39b3ccb09d9cdef6888a40ada172b68408f65f69808ca033dc2108dd9e99765`로
  등록값 및 실제 파일 해시가 일치했다. E0를 재학습하거나 resume하지 않는다.
- 보정본의 모든 이미지·라벨·분할 artifact 해시를 재검증했고 fingerprint
  `c3d63dfa73314a36b5522f4651f5fd7d784bd8a054e78916f821160729862e2b`가 일치했다.
  고정 COCO 모델의 크기와 SHA256도 일치했다.
- 비교 기준은 서버 E0 공통 Validation mAP50-95 0.4445945004,
  confidence 0.25 Fire Recall 0.6949860724, Negative 오탐 이미지 27/1,610이다.
  세부 측정 근거는 [기존 E0 보고서](007_remote_e0_baseline.md)에 있다.
- 독립 큐의 승인 범위는 E1 → E2-A → E2-C → E3-HSV → E3-Mosaic → E4-800이다.
  각 단계의 학습·무결성 검사·공통 Validation·비교 보고가 통과해야 다음 단계를 시작한다.
  모든 조건은 COCO에서 새로 시작하며 E3의 기준 lr0는 0.01로 유지한다.
- 작업 관리자는 독립 세션, OS 잠금, 원자적 JSON, 정확한 run 연결, checkpoint 기반
  진행 조회와 유한 실패 정책을 갖추어 검증한다. 설정·입력·checkpoint 불일치 및 OOM은
  큐를 중단한다. 자동 batch/AMP 조정이나 무한 재시도는 하지 않는다.
- E5는 선택 실험이며 자동 실행하지 않는다. 핵심 큐 종료 후 `awaiting_analysis`로
  남기고 E6 설정·최종 선택·Test는 실제 Validation 분석 후 진행한다.

준비 시점에는 E1을 시작하지 않았다. 구현 검증과 실제 실행 기록은 별도 보고서에 남긴다.
