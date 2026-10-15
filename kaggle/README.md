# Kaggle GPU 환경

GPU가 없는 개발 환경 대신 Kaggle 노트북(GPU T4×2, 주당 30시간)에서 학습·측정하기 위한 준비물이다.
소스는 비공개 데이터셋으로 올리고, 커널은 그 데이터셋을 입력으로 받아 설치한 뒤 돈다.

| 파일 | 하는 일 |
|---|---|
| `push_source.sh` | 저장소 소스(커밋·추적 파일, `runs/`의 체크포인트 제외)를 비공개 데이터셋 `irukidev/stream-mind-src`로 올린다. 어떤 커밋인지 `kaggle/SOURCE.json`에 남긴다 |
| `run_kernel.sh` | 커널 폴더를 올려 실행하고, 끝날 때까지 기다린 뒤 출력을 `kaggle/.build/out/<커널>/`로 받는다 |
| `gpu_check.py` | GPU 연기 시험. 커널 수치(조각 = 순환, cpu = cuda), 작은 SFT, 저장·불러오기, 어댑터, TorchBackend(대화·게임·포크 후보 점수), RL 한 번 |
| `smoke/` | 연기 시험 커널: 소스 설치 → 단위 테스트(CPU) → `gpu_check.py`(GPU). 학습하지 않는다 |
| `train/` | 학습 커널 틀: `scripts/pipeline.sh`를 GPU로 돌리고 `runs/`를 커널 출력에 남긴다 (다음 단계) |

## 쓰는 법

```bash
# KAGGLE_API_TOKEN 이 환경 변수로 설정되어 있어야 한다 (kaggle CLI 2.x)
pip install kaggle
bash kaggle/push_source.sh                 # 소스를 바꿀 때마다 먼저
bash kaggle/run_kernel.sh kaggle/smoke     # GPU 연기 시험 (몇 분)
bash kaggle/run_kernel.sh kaggle/train     # 학습 (다음 단계)
```

- 장치는 `STREAM_MIND_DEVICE`(cpu | cuda | cuda:N | auto) 또는 CLI `--device`로 고른다. 기본은 cpu다.
- `scripts/pipeline.sh`는 GPU가 2장이면 stream·turn 병렬 학습을 `cuda:0`, `cuda:1`에 나눠 싣는다.
- 커널은 모두 비공개(`is_private: true`)이고, 기계는 `machine_shape: NvidiaTeslaT4`(T4×2)다.
  `NvidiaL4` 등으로 바꿀 수 있다. Kaggle GPU 세션은 한 번에 최대 12시간이다.
- 소스 데이터셋은 압축 파일로 올리지만 Kaggle이 풀어 둔다. 커널은 두 경우를 모두 찾는다.
