# CAN 제어 Jetson 포팅 (Kvaser → SocketCAN)

현재 `depth_cam/calib/control.py` 는 **Kvaser canlib** 기반입니다. Jetson(ARM)에서는
Kvaser 드라이버가 안 될 가능성이 높아, 리눅스 표준 **SocketCAN**(`python-can`) 으로
포팅하거나 Kvaser ARM 지원을 확인해야 합니다. canlib 부재 시 코드는 자동 **MOCK**
(FSM 시각화만, 리프터 실제 안 움직임).

## 확인 먼저
```bash
python jetson_fsm_deploy/preflight.py    # "CAN 제어" 항목이 MOCK 인지 확인
```
MOCK 이면 아래 둘 중 하나.

## 옵션 A — Kvaser ARM (하드웨어가 Kvaser면)
Kvaser Linux SDK(canlib) ARM/aarch64 빌드가 되는지 확인. 되면 현 코드 그대로 동작.

## 옵션 B — SocketCAN 포팅 (USB-CAN 어댑터 / Jetson 내장 CAN)
1. CAN 인터페이스 활성화:
   ```bash
   sudo ip link set can0 type can bitrate 500000   # bitrate 는 리프터 사양에 맞게
   sudo ip link set up can0
   ```
2. `pip install python-can`
3. `control.py` 의 송신부를 python-can 으로 교체 (인터페이스만 바꾸고 frame ID/data/주기는 유지):
   - `canlib ... ch.write(Frame(id, data, flags))` → `bus.send(can.Message(arbitration_id=id, data=data, is_extended_id=...))`
   - `can_init/can_close` → `bus = can.interface.Bus(channel='can0', bustype='socketcan')` / `bus.shutdown()`
   - heartbeat 송신 주기/CAN ID/바이트 포맷은 **현 코드 값 그대로** 유지 (리프터 펌웨어 약속).

> ⚠️ frame ID·data 바이트·bitrate·확장ID 여부는 **리프터 측 약속**이라 절대 바꾸지 말 것.
> 인터페이스(드라이버)만 Kvaser→SocketCAN 으로 교체.

## 안전
- 첫 실보드 테스트는 리프터를 **들어올린 채/바퀴 띄운 채** 또는 비상정지 손에 두고.
- `STOP_SEC=1.2` 인터록이 상태 전이 사이 정지를 보장하지만, 잘못된 pose가 들어가면
  잘못된 회전 cmd가 갈 수 있으므로 **GATE_PROFILE=real** 로 나쁜 프레임을 먼저 거른다.
