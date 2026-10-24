# C 독립 visible-part witness

고정 8사례의 CLEAN96/VALID96 16개와 같은 세 프레임의 원 fullRGB/q8 overlay/contact 9개, 총 25개의 허용 PNG를 이번 V206에서 직접 toolview했다. 이전 V205의 관찰이나 A의 새 ROI를 대체 근거로 삼지 않았다. A 새 ROI 값은 읽지 않았다.

8개 모두 원 crop 안의 노출된 상판 끝·블록 모서리를 식별해 약한 정수 inclusive ROI를 기록했다. 잘린 앞 오른쪽 사례는 padding 아래로 부위를 복원하지 않고 유효 상부의 노출 조각만 잡았다. 박스는 visible feature 주변 직사각형으로 일부 배경을 포함할 수 있으며, 부품 전체 분할·3D 점·corrected q가 아니다. q 중심 포함, 최소 면적, 완벽한 mesh 또는 위치 정답을 요구하지 않았다.

범용 ROI validator와 별도로 원 8행 순서/ID/qL/native/가용성/prior flags/fitfalse 및 각 행의 same-case CLEAN96와 same-frame full context 실제 view 바인딩을 검증했다. 원본 q/L/native와 V205 strict 결과는 바꾸지 않았다. 물리 native gold·합의·학습 허가·새 E 라벨은 생성하지 않았다. 이후 root/B가 양쪽 봉인 뒤 의미/overlap을 검토한다.
