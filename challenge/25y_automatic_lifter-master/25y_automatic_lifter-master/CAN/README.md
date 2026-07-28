# 🚜 DirectFrameForkliftController

**무인 지게차를 노트북 키보드로 원격 조종**하기 위한 **CAN 통신 기반 제어기** 
Kvaser CANlib를 활용하여 지게차의 **주행 / 회전 / 리프트 / 폴딩 / 리치 모드**를 안전하게 제어할 수 있도록 설계

---

## 📊 주요 기능

- **CAN 통신 기반 제어**
  - Kvaser CANlib + Frame 사용  
  - 표준(11-bit) 및 확장(29-bit) ID 전송 지원  

- **Heartbeat 기반 안정화**
  - 200ms 주기 Heartbeat 전송으로 워치독 타임아웃 방지  
  - 시작/전환 시 안정화 버스트 전송 → 수신기 활성 보장  

- **딸깍(릴레이 잡음) 방지 최적화**
  - Idle 상태에서도 movement/control 프레임을 주기적으로 유지 전송  
  - Active/Idle 모드별 전송 주기 차등 적용  

- **모드 전환 및 다기능 제어**
  - Driving / Lift / Folding / Reach 모드 지원  
  - 키보드 입력(WASD, Q/E, U/I/O/P, K/L 등)으로 다양한 동작 수행  
  - Emergency Stop/Release 제공 (SPACE/M 키)  

- **안전 메커니즘**
  - Emergency Stop 시 제어 프레임 버스트 전송  
  - Neutral Guard: Stop → Non-stop 전환 시 지연 적용  
  - HOLD_GRACE: 키 입력 해제 후 200ms간 유지  

---

## 🎮 조작 방법

- **주행 모드 (Driving)**
  - `W/A/S/D`: 전진 / 좌회전 / 후진 / 우회전  
  - `Q/E`: 제자리 반시계 / 시계 회전  
  - `W+A`: 전진+좌회전, `S+D`: 후진+우회전 등 복합 입력 가능  

- **모드 전환**
  - `U`: Driving 모드  
  - `I`: Lift 모드 (K=올림, L=내림)  
  - `O`: Folding 모드 (K=전개, L=접기)  
  - `P`: Reach 모드 (K=전진, L=후진)  

- **비상 제어**
  - `SPACE`: Emergency Stop (비상정지)  
  - `M`: Emergency Release (해제)  

- **종료**
  - `Z`: 프로그램 종료  

---

## ⚙ 실행 방법

```bash
python control_forklift_v2.py
