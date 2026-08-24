# dpvla_2026

DPVLA(Dual-Process VLA) 연구·배포 코드 모음 — 학습(starVLA)·실로봇 추론(unitree G1)·모니터링 GUI.

## 구성

### `starvla_2026/` — starVLA 작업본 (DPVLA 학습)
공식 [starVLA](https://github.com/starVLA/starVLA) 스냅샷(`starVLA_dev @ 0ed0aad2`) 위에
DPVLA-DPFM(×QFormer_V1) 학습 코드를 적용한 트리.

| 경로 | 역할 |
|---|---|
| `starVLA/model/framework/VLM4A/DPVLA_QFormer_V1.py` | DPVLA 프레임워크 본체 (System-2 latent → readout 토큰 방식) |
| `starVLA/model/framework/VLM4A/DPVLA.py` | 부모 클래스 (QFormer_V1이 상속) |
| `starVLA/model/modules/action_model/DPFM_ActionHeader.py` | DPFM 액션헤드 — Flow-Matching DiT (`head_type: dit`) |
| `starVLA/model/modules/action_model/typed_latent_qformer.py` | latent Q-Former 모듈 (프레임워크가 import) |
| `starVLA/model/modules/dino_model/dinov3_encoder.py` | System-1 비전 (DINOv3-ViT-B/16) |
| `starVLA/model/modules/vlm/QWen3_5.py` | System-2 (Qwen3.5) 인터페이스 — readout 토큰 `<\|dpvla_ro_i\|>` 등록·부착 |
| `starVLA/training/train_starvla.py` + `starVLA/dataloader/*` | 학습 루프·데이터로더 (sys1_hz/sys2_hz 샘플링) |

> ⚠️ 미적용분: 유효배치 512용 deepspeed accum 설정(`deepspeed_zero2_qfv1_8gpu_accum4.yaml` 쌍),
> 챔피언 학습 yaml(`starvla_dpvla_qfv1p8_4b_fm_ro_robotpretrain_*.yaml`), 런처 스크립트.
> 전체 원본·문서·개발일지는 `shinjaeho980610/starvla_dpvla_2026` 참조.

### `unitree_lerobot/` — Unitree G1 추론용 lerobot 스택
공식 [unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot) 스냅샷(`main @ 41c28057`).

| 경로 | 역할 |
|---|---|
| `unitree_lerobot/lerobot/` | 내장 huggingface/lerobot **v0.4.1** (`a5b29d43` — 공식 submodule 핀과 동일 커밋을 실파일로 동봉, `--recurse-submodules` 불필요) |
| `unitree_lerobot/eval_robot/` | ★**DPVLA 실로봇 평가 작업본으로 교체됨** — `eval_g1_dp*.py`(dual-process G1 평가 시리즈), `g1_inference_ui.py`(추론 UI), `robot_control/`(팔 IK + brainco/inspire/unitree 핸드 제어), `image_server/`(카메라 서버) |
| `data_editor/` · `docs/` · `test/` | 공식 데이터셋 편집기·문서·테스트 |

> 공식 원본 eval_robot(brainco URDF 자산, UniArmL1 지원, 신판 image_server)은 커밋 이력
> (`d51879c` 직전)에서 복구 가능.

### `g1_gui_code/` — G1 추론 모니터링 GUI
웹 기반 G1 추론용 GUI 프로그램 (python -m 실행형).

| 파일 | 역할 |
|---|---|
| `server.py` + `__main__.py` | GUI 백엔드 서버 |
| `static/` (app.js·style.css) | 웹 프론트엔드 |
| `realsense_streamer.py` / `mjpeg_streamer.py` | 카메라 영상 스트리밍 |
| `evaluation_protocol_config.json` | 평가 프로토콜 설정 |
