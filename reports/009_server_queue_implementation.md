# 독립 학습 큐 구현 및 검증

2026-11-04 UTC. 승인된 핵심 실험의 장시간 실행을 Codex/SSH와 분리하기 위해
`scripts/training_queue.py`와 `src/fire_detection/queue.py`를 추가했다.
이 보고서의 단위 테스트·probe는 실제 모델 실험 결과가 아니다.

## 실행 및 실패 정책

- 유한 순서: E1, E2-A, E2-C, E3-HSV, E3-Mosaic, E4-800.
- 기존 학습·공통 Validation API를 재사용한다. 생성 직후 callback으로 정확한 run을
  receipt에 기록해 부모에게 전달한다. 완료 폴더의 시각으로 경로를 추측하지 않는다.
- 관리자와 각 작업자는 독립 세션으로 실행한다. stdin은 `/dev/null`, 출력은 `work/` 로그,
  SIGHUP은 무시한다. OS 파일 잠금과 PID 생성 시각을 사용하고 관리자 잠금을 작업자에게
  상속해 관리자만 종료된 경우에도 중복 시작을 막는다. 현재 Codex tmux는 유지한다.
- 원자적 상태 JSON에 계획/현재 단계/다음 작업, run 경로, PID/생성 시각, 마지막 저장
  epoch, 시작/종료 시각과 실패 이유를 보존한다.
- COCO 모델·확장 설정·데이터 fingerprint·소스/설정 SHA256 및 E0 기준 기록을 고정한다.
  학습 완료·best/last 해시·epoch·평가 모델/입력/설정 연결과 전체 입력 검증 후에만 다음
  단계로 진행한다. 공통 Validation과 운영 confidence 0.25 지표는 구분해 보고한다.
- OOM/무결성/일반 실패는 전체 큐를 중단한다. 자동 재시도·batch 축소·AMP 변경은 없다.
  명시적 일시 중단 재개만 같은 run/설정과 검증된 `last.pt`를 사용한다.
- 각 완료 단계에 한국어 보고서와 비교 CSV/JSON·학습 곡선을 생성한다. 마지막에는
  `awaiting_analysis`로 멈추며 선택 실험/E6 override/Test를 추측해 실행하지 않는다.

## 검증

`.venv/bin/python -m unittest discover -s tests -v`의 기존 회귀 및 새 큐 테스트 50개를
통과했다. 별도 probe PID 4868은 launcher 종료 후 PID 1 아래에서 독립 세션 SID 4868로
계속 실행됐으며 SIGHUP 이후에도 정상 완료했다. stdin은 `/dev/null`, 출력은
`work/server-detach-probe.log`였다. 이 검증은 실제 SSH 연결을 끊는 대신 부모 종료와
SIGHUP을 재현했으며 학습/GPU를 사용하지 않았다. 동명 JSON에 근거를 보존한다.
`.venv/bin/python -m pip check`에서도 의존성 충돌이 없음을 확인했다.
Linux 전용 큐 테스트는 Windows에서 해당 모듈을 import하기 전에 건너뛰도록 처리했다.
수정 후 Linux 전체 50개를 다시 통과했다. 플랫폼 분기 모의 점검은 통과했지만
Windows 네이티브 실행을 검증한 것은 아니다. 학습 중 production 소스는 변경하지 않았다.

핵심 검증은 중복 잠금 거부, PID 재사용 구분, launcher 종료 및 SIGHUP 이후 probe 완료,
등록 checkpoint 변조 거부, 학습 조건 drift 거부, 다른 run의 평가 연결 거부,
기록 없는 run 추측 거부, OOM 재개 거부, 각 단계의 Validation gate 및
E6 분석 대기, 실패 후 다음 조건 미시작이다.

실제 E1의 학습 진행 및 checkpoint 관측은 구현 검증과 분리해 기록한다.
사용·로그·상태 조회·복구 명령은 [USAGE](../docs/USAGE.md)에 있다.
