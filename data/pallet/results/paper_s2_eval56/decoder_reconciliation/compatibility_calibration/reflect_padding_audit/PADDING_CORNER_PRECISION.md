# Where the recovered corners actually land

Newly recovered corners only: raw peak <= 0.30 without padding, > 0.30 with it.

```
        arm  new  median px  <=10px  <=20px  <=50px  >50px  >100px
──────────────────────────────────────────────────────────────────
    reflect   50       38.2     12%     20%     64%    36%     12%
  replicate   66       39.4     21%     27%     59%    41%     18%
constant127   70       41.0     17%     36%     60%    40%     16%
```

The gate asks for at least 60% within 20px and at most 15% beyond 50px.  The
best arm reaches **36% and 40%**.  Response recovery is not localisation.

## The split that explains it

```
        arm  in-frame n  median px  <=20px  off-screen n  median px  <=20px
───────────────────────────────────────────────────────────────────────────
    reflect          43       36.9     23%             7      254.6      0%
  replicate          56       37.4     32%            10      295.0      0%
constant127          53       21.7     47%            17      286.3      0%
```

Corners whose GT is **inside** the original frame come back at a median of
25-30px -- mediocre but real.  Corners whose GT is **outside** the frame come
back at a median of roughly **290px**: the model puts something confident
somewhere, and it is not near the true off-screen position.

That is the whole of the localisation failure.  Padding tells the network a
pallet is present and roughly where its visible part is; it cannot tell it where
a corner it has never been supervised on should go.  Under the current
supervision an off-screen corner has no target, so the recovered response for
those channels is unanchored.

`figures/corner_precision_after_padding.png` shows the two distributions side by
side; `figures/dead_response_before_after.png` marks off-screen recoveries in
red.
