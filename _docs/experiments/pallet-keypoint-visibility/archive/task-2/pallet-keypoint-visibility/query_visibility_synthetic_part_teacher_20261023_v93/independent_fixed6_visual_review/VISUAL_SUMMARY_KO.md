# V93 고정6시점 geometry 시각 검토

**실제6PNG 전부를 열어 검토했습니다.** 평행한 deck판·가로 받침·블록·포크 통로가 같은 얕은 직사각 구조를 이뤄 한 팔레트형 조립체라는 시각적 근거가 있습니다. 원본PS에서 명확한 별도물체/두번째팔레트는 보이지 않았으나308개 tinyprim의 개별소유까지 인증한 것은 아닙니다.

선택part152는 plusY에서 한쪽 가장자리 근처의 **길고 좁은 상부 deckboard 한 장 후보**로 보입니다. foot/corner자체가 아니며, 다른시점에서는 매우 얇은edge로 보입니다. 이는 node이름만의 추측을 넘어선 조건부AI crossview관찰이지만 authorverifiedpart나 native8gold는 아닙니다.

| 고정 시점 | 관찰 |
|---|---|
| minusX | selectedalone edge가 PS에서는 숨지만 다른부재와 통로는 남음. PVforeign은 원래selfhidden과 구분해야 함 |
| minusY | 선택판 일부가 가로부재 뒤로 숨고, foreign은 남은판일부를 덮음. 다른판은 계속 보임 |
| minusZ | selected가 얇고 일부만 보임. foreign이 일부가려도 wholepallet은 보임 |
| plusX | selectedsupport가 PS에서 사라져도 팔레트 구조는 보임 |
| plusY | 긴 상부판 모양이 가장명확. foreign이 대부분덮어도 옆deck판은 계속보임 |
| plusZ | PS의 남은 selectedsupport가 foreign뒤로숨지만 다른팔레트면은 보임 |

**전체팔레트/재질지원과 기대part지원은 다릅니다.** 이 sixview는 partID별teacher가 다른판의가시성으로 expectedpart를D로승격하는오류를 막을 수 있다는 geometry개연성을 보입니다. 그러나 wholeboardfraction은 특정끝점·접합선·native8reference 가시성의 정답이 아닙니다. native8연결·humanD/E/S·학습admission은 계속UNKNOWN/false입니다.

진단색은실제재질RGB가 아니고, 빨간foreign308은인위적geometry입니다. 원그림의proxycounts/legend를본상태의 독립AI시각개연성검토이며 blindgold/새raster수치감사가 아닙니다. MissingMDL·실제물리scale·sourceauthor/license 문제도 그대로입니다. 모델·fit·current·PnP·추가render·좌표변경·새라벨0입니다.
