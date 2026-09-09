# DOPE / YOLO plus DHT integration verification

Compare frozen baseline corner predictions against inference-time fusion with existing DHT8 structural side lines. Select fusion strength on synthetic validation only; evaluate unchanged settings on synthetic held-out images and canonical real DEV52, including coverage, corner and line error, failures, actual latency and qualitative cases. No real training, GT-driven inference, final-test access or new padding experiment runs. A negative result is a valid result. This does not claim joint training or a native YOLO DHT head.
