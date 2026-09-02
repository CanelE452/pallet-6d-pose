# Daytime visibility audit

생성 2026-09-02.  MAIN Daytime 에서 strict keypoint 지표가 왜 비는지 **직접 세어** 확인한다.

## 대상

```text
paper_domain == daytime  AND  object_type == plastic  AND  PAPER_EVAL member
```

```text
frames                       103
keypoints                    927
visibility unknown           927
  그중 화면 안 (실제 작업량)           895
  그중 화면 밖                    32
```

세션별:

```text
eval_cad             18
eval_night08         12
eval_night09         16
eval_noapril         12
wood_183705          25
wood_184309          20
```

## 원인 분류 — 추측하지 않고 센다

```text
A  xy 존재 + visibility unknown      사람이 visibility 만 채우면 된다
B  xy 자체 없음                      좌표 작업이 필요하다
C  schema 문제                       파일을 고쳐야 한다
D  annotation 파일 없음              어노테이션이 없다
```

```text
A_XY_BUT_VISIBILITY_UNKNOWN        103
```

## 판정

전부 A 다.  **좌표를 다시 찍을 일이 없다** — 필요한 작업은 895 개
keypoint 의 visibility 라벨뿐이다.  화면 밖 점은 visibility 0 이 맞으므로 건드리지 않는다.

이 판정이 review 범위를 정한다.  좌표 편집은 도구에서 막는다.
