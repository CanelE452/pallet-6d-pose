Lift FSM Birdview Simulator
===========================

실행 방법
1. 가장 간단한 방법: run_simulator.bat 실행
   - 기본 브라우저에서 index.html을 바로 엽니다.

2. 로컬 서버 방식: run_local_server.bat 실행
   - http://127.0.0.1:8765/ 로 실행됩니다.
   - PowerShell만 있으면 됩니다.
   - 종료하려면 서버 창에서 Ctrl+C를 누르세요.

포함 파일
- index.html
- styles.css
- sim.js
- simulator_preview.png
- run_simulator.bat
- run_local_server.bat
- serve.ps1

후진 거리 조절
- UI 오른쪽 Motion Model의 Reverse sec 값을 조절합니다.
- 시뮬레이터 후진 거리 = Move m/s * Reverse sec 입니다.

