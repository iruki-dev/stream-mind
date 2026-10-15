# Stream-Mind (가칭)

하나의 끝나지 않는 스트림 위에서 입력을 받으면서 생각·출력·행동하는 텍스트 언어 모델을 만드는 프로젝트다.
이 저장소는 설계안 로드맵 P0~P4 전체를 담고 있다. P0은 모델 없이 오케스트레이터(이벤트 구동, 상태 포크,
도구 규격, 위험 등급)를 검증하고, P1~P4는 설계안의 하이브리드 백본을 직접 구현한 소형 모델(StreamLM)을
스트림 형식으로 학습하고, 기억을 결합하고, 쉬는 시간에 스스로 공부하게 하고, 환경 RL을 한 뒤 관문마다 잰다.

P0 설계 대응과 측정은 [docs/p0.md](docs/p0.md), P1~P4는 [docs/p1-p4.md](docs/p1-p4.md)에 있다.

## 지금 상태

| 단계 | 내용 | 관문 |
|---|---|---|
| P0 | 오케스트레이터(이벤트 구동, 상태 포크, 공통 도구 규격, 위험 등급) | 통과 |
| P1 | StreamLM(Gated DeltaNet + sink·윈도 어텐션, 362만 매개변수)을 스트림 형식으로 SFT | 미달 (정적 −0.04, Snake-lite) |
| P2 | 압축을 켠 채 학습, 노트 도구 연결 | 통과 (지연·메모리 일정, 장기 회상 유지) |
| P3 | 유휴 학습(공고화, 실패 복습, 도구 연습, 질문 준비, 자기 출제, 공부법 밴딧), 어댑터 버전·삭제 | 통과 |
| P4 | 환경 RL(게임·대화), 안전(오염 추적) | 미달 (RL 후보 거부, Snake-lite) |

GPU 없이 CPU 4코어에서 학습한 소형 모델의 결과다. 설계안의 0.5~2B 모델은 이 환경에서 학습할 수 없다.
목표 G1~G10 판정은 [docs/results.md](docs/results.md), 방법과 해석은 [docs/p1-p4.md](docs/p1-p4.md)에 있다.
학습한 체크포인트(`runs/`)는 저장소에 넣지 않았다. 측정 결과 JSON은 `runs/results/`에 있다.

## 들어 있는 것

| 설계안 | 구현 | 파일 |
|---|---|---|
| 모델과 도구의 경계, 공통 도구 규격 (설명서·호출·관측·알림·신뢰 등급) | `Tool` 규격 | `src/stream_mind/tools/base.py` |
| 도구 4개: 메신저, 터미널, 게임, 노트 | 각 도구 | `src/stream_mind/tools/` |
| 핵심 모듈 2 · 영속 스트림 (`<obs>`, `<think>`, `<call>`, `<wait>`, `[T=t]`) | `PersistentStream` | `src/stream_mind/stream.py` |
| 핵심 모듈 2 · 이벤트 구동 깨우기, 사고 예산, 미해결 질문 기록 | `Mind.step` | `src/stream_mind/mind.py` |
| 핵심 모듈 3 · 상태 포크 (분기에서만 행동, 원 스트림은 실행 기록을 관측) | `Mind._branch` | `src/stream_mind/mind.py` |
| 핵심 모듈 4 · 닫힌 행동 공간은 후보 점수 매기기, 마감까지 없으면 기본 행동 | `ActionRequest`, `GameTool` | `src/stream_mind/tools/` |
| 핵심 모듈 5 · 행동 위험 등급 0/1/2, 내용 모르는 실행은 등급 2, 샌드박스 한 단계 하향 | `classify_shell`, `RiskGate` | `src/stream_mind/risk.py` |
| 실시간 텍스트 게임 (Freeway, Snake) | 멈추지 않는 환경 | `src/stream_mind/envs/` |
| 턴 방식 비교 기준 | `TurnAgent` | `src/stream_mind/baselines.py` |
| 관문 1 측정 | 벤치마크 | `src/stream_mind/bench/gate1.py` |
| 소형 하이브리드 체크포인트 연결 | `HFBackend` | `src/stream_mind/model/hf.py` |
| 백본 (Gated DeltaNet + sink·윈도 어텐션, 어댑터) | `StreamLM` | `src/stream_mind/nn/` |
| 스트림 규약, 학생 백엔드 | `TextStreamBackend`, `TorchBackend` | `src/stream_mind/model/` |
| P1 데이터(교사, 누설 검사), SFT | `TeacherBackend`, `train/sft.py` | `src/stream_mind/data/`, `src/stream_mind/train/` |
| P3 유휴 학습, 어댑터 저장소 | `IdleLearner`, `AdapterBank` | `src/stream_mind/idle/` |
| P4 환경 RL | `rl_train` | `src/stream_mind/train/rl.py` |
| 관문 2~5, G6·G8·G10 측정 | 평가기 | `src/stream_mind/bench/gates.py` |

## 빠른 시작

```bash
pip install -e '.[dev]'                    # torch 포함
pytest -q                                  # 단위 테스트
python -m stream_mind.bench.gate1 --seeds 20   # 관문 1
```

실제 모델(예: `Qwen/Qwen3.5-0.8B`, Gated DeltaNet 하이브리드)로 연기 시험:

```bash
pip install -e '.[hf]'
python examples/hf_smoke.py --model Qwen/Qwen3.5-0.8B --seconds 60
```

전체 학습·측정 순서는 `scripts/pipeline.sh` (CPU에서 약 12시간).

## 관문 1 결과 (요약)

시뮬레이션 백엔드, 시드 20개, 100프레임. 칸은 평균 점수.
시간 압박 = 사고가 끝까지 수렴하는 시간 / 프레임 간격.

| 게임 | 압박 | stream | turn | stream − turn |
|---|---|---|---|---|
| freeway | 0.25 | 7.60 | 7.60 | +0.00 |
| freeway | 2 | 7.60 | 3.45 | +4.15 |
| freeway | 4 | 7.60 | 0.35 | +7.25 |
| snake | 0.25 | 16.90 | 16.90 | +0.00 |
| snake | 2 | 16.80 | 7.50 | +9.30 |
| snake | 4 | 15.50 | 2.40 | +13.10 |

사고가 한 프레임 안에 끝나면 두 방식이 같고, 사고가 프레임을 넘기기 시작하면 stream이 턴 방식을
크게 앞선다. **관문 1 통과.** 다만 이 결과는 언어 모델이 아니라 "생각할수록 판단이 좋아지는"
계획기로 측정한 것이라, 검증한 것은 스케줄링과 포크의 제어 논리다. 전체 표와 한계는 docs/p0.md 참고.
