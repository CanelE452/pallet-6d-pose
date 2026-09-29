# 반복43 사전검사 환경·자원 원장 주석

첫CPU 사전검사는 sandbox에서workers2의PyTorch tensor IPC socket공유가 `PermissionError: [Errno 1] Operation not permitted`로차단되었다. 해당실행세션62417만Ctrl-C로종료했다. 새fit·optimizer update·GPU실행은0이다. 별도private오류기록을보존했다.

같은고정코드·프로토콜·workers2·seed·sampler를host IPC권한으로재실행했다. 이는환경오류재실행이지성능에따른알고리즘재시도가아니다. 다른작업프로세스나환경을변경하지않았다. 최종결과는 `PREFLIGHT.json`을확인한다.

`PRIMARY_PROTOCOL.json`의 `global_resource_ledger` 해시는반복학습시작전의원장을가리킨다. 최상위원장은이후fit마다정상적으로변한다. 감사용동일바이트사본을 [RESOURCE_LEDGER_PREFIT_S43.json](RESOURCE_LEDGER_PREFIT_S43.json)에보존했다. 사전SHA는 `4b71e71691bed09290931f61d4f530614610c460ac879182747954bb655bd5df`다.

사전누적은학생4fit·1280update·학습GPU281.13820330007초·selector fit0이다. 반복2fit도최상위 `RESOURCE_LEDGER.json` 한곳에 `REPEAT_PRIMARY_S43::` 접두사event로추가하며stage별0원장을만들지않는다. 고정프로토콜은수정하지않았다.
