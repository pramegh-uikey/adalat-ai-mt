# Error analysis

## Per-system fidelity

| System | Numbers | Citations | Terms (any/ref) | Repetition | Omission | Addition | Mean length ratio |
|---|---|---|---|---|---|---|---|
| IndicTrans2-1B zero-shot | 93.6% | 100.0% | 82.5%/57.9% | 1 | 3 | 0 | 0.96 |
| IndicTrans2-dist-200M zero-shot | 96.2% | 100.0% | 89.7%/65.9% | 0 | 2 | 0 | 0.96 |
| NLLB-200-distilled-600M zero-shot | 96.2% | 100.0% | 56.3%/31.0% | 0 | 5 | 0 | 0.95 |
| IndicTrans2-1B + LoRA (legal) | 95.5% | 100.0% | 96.8%/71.4% | 1 | 2 | 1 | 0.99 |
| IndicTrans2-dist-200M + LoRA (legal) | 92.3% | 100.0% | 98.4%/81.7% | 1 | 1 | 1 | 1.01 |

## Robustness: corpus chrF++ on segments with >= N source words

| N | segments | IndicTrans2-1B zero-shot | IndicTrans2-dist-200M zero-shot | NLLB-200-distilled-600M zero-shot | IndicTrans2-1B + LoRA (legal) | IndicTrans2-dist-200M + LoRA (legal) |
|---|---|---|---|---|---|---|
| 0 | 120 | 64.42 | 65.27 | 55.39 | 67.43 | 66.94 |
| 5 | 114 | 64.50 | 65.43 | 55.37 | 67.32 | 66.83 |
| 10 | 106 | 64.72 | 65.71 | 55.66 | 67.54 | 67.04 |

## IndicTrans2-1B + LoRA (legal) vs IndicTrans2-1B zero-shot

Overall: win=57 tie=36 loss=27

| Source length | Win | Tie | Loss |
|---|---|---|---|
| <=20 | 19 | 15 | 6 |
| 21-40 | 27 | 12 | 14 |
| >40 | 11 | 9 | 7 |

| Term category | Win | Tie | Loss |
|---|---|---|---|
| archaic/formulaic | 6 | 0 | 3 |
| legal-term | 41 | 25 | 24 |
| named-entity | 6 | 9 | 2 |
| citation/section/date | 6 | 3 | 1 |

Top gains: 10-0000 (+86.3), 11-0000 (+80.5), 27-0021 (+64.1), 27-0000 (+41.4), 27-0003 (+22.0), 11-0007 (+20.4), 11-0009 (+20.1), 11-0020 (+18.8), 27-0030 (+18.2), 11-0023 (+18.2)

Top losses: 27-0007 (-19.9), 27-0016 (-17.1), 11-0018 (-13.6), 27-0010 (-12.4), 27-0027 (-11.5), 10-0022 (-10.7), 10-0048 (-10.1), 10-0023 (-9.9), 10-0006 (-9.6), 27-0014 (-9.2)

## IndicTrans2-dist-200M + LoRA (legal) vs IndicTrans2-dist-200M zero-shot

Overall: win=53 tie=44 loss=23

| Source length | Win | Tie | Loss |
|---|---|---|---|
| <=20 | 19 | 18 | 3 |
| 21-40 | 26 | 16 | 11 |
| >40 | 8 | 10 | 9 |

| Term category | Win | Tie | Loss |
|---|---|---|---|
| archaic/formulaic | 5 | 3 | 1 |
| legal-term | 34 | 31 | 25 |
| named-entity | 6 | 8 | 3 |
| citation/section/date | 2 | 7 | 1 |

Top gains: 10-0000 (+86.3), 11-0000 (+80.5), 27-0000 (+68.4), 27-0041 (+67.2), 11-0006 (+23.7), 27-0016 (+21.8), 27-0030 (+21.3), 11-0007 (+20.1), 11-0019 (+18.1), 27-0040 (+16.9)

Top losses: 10-0018 (-21.9), 27-0027 (-19.3), 27-0007 (-13.8), 27-0019 (-13.7), 27-0004 (-12.6), 11-0015 (-9.0), 27-0033 (-8.9), 10-0032 (-8.3), 10-0006 (-7.9), 11-0005 (-7.5)

## IndicTrans2-dist-200M zero-shot vs IndicTrans2-1B zero-shot

Overall: win=34 tie=45 loss=41

| Source length | Win | Tie | Loss |
|---|---|---|---|
| <=20 | 8 | 19 | 13 |
| 21-40 | 21 | 15 | 17 |
| >40 | 5 | 11 | 11 |

| Term category | Win | Tie | Loss |
|---|---|---|---|
| archaic/formulaic | 1 | 5 | 3 |
| legal-term | 30 | 30 | 30 |
| named-entity | 6 | 7 | 4 |
| citation/section/date | 0 | 3 | 7 |

Top gains: 27-0021 (+62.9), 11-0023 (+22.5), 10-0042 (+13.7), 10-0030 (+11.0), 11-0014 (+11.0), 10-0013 (+10.4), 10-0028 (+9.8), 11-0005 (+9.6), 11-0011 (+9.0), 10-0002 (+8.8)

Top losses: 27-0041 (-67.2), 27-0000 (-27.0), 27-0016 (-21.8), 27-0011 (-20.9), 10-0022 (-14.7), 11-0017 (-14.4), 27-0020 (-14.4), 10-0031 (-14.1), 27-0017 (-12.8), 11-0006 (-10.2)

## NLLB-200-distilled-600M zero-shot vs IndicTrans2-1B zero-shot

Overall: win=18 tie=19 loss=83

| Source length | Win | Tie | Loss |
|---|---|---|---|
| <=20 | 8 | 7 | 25 |
| 21-40 | 6 | 8 | 39 |
| >40 | 4 | 4 | 19 |

| Term category | Win | Tie | Loss |
|---|---|---|---|
| archaic/formulaic | 2 | 1 | 6 |
| legal-term | 13 | 10 | 67 |
| named-entity | 2 | 2 | 13 |
| citation/section/date | 0 | 1 | 9 |

Top gains: 27-0021 (+48.6), 10-0036 (+24.1), 11-0007 (+23.9), 10-0000 (+21.0), 11-0009 (+19.7), 10-0037 (+18.3), 11-0000 (+17.2), 11-0023 (+14.4), 27-0008 (+13.2), 11-0024 (+10.0)

Top losses: 10-0013 (-71.6), 10-0012 (-52.2), 27-0020 (-47.0), 27-0010 (-44.3), 27-0016 (-43.9), 10-0017 (-43.0), 27-0028 (-35.0), 27-0023 (-30.4), 10-0030 (-29.6), 27-0026 (-26.9)

## IndicTrans2-dist-200M + LoRA (legal) vs IndicTrans2-1B + LoRA (legal)

Overall: win=45 tie=35 loss=40

| Source length | Win | Tie | Loss |
|---|---|---|---|
| <=20 | 13 | 16 | 11 |
| 21-40 | 27 | 12 | 14 |
| >40 | 5 | 7 | 15 |

| Term category | Win | Tie | Loss |
|---|---|---|---|
| archaic/formulaic | 2 | 4 | 3 |
| legal-term | 34 | 25 | 31 |
| named-entity | 7 | 6 | 4 |
| citation/section/date | 1 | 3 | 6 |

Top gains: 27-0016 (+17.1), 27-0040 (+16.9), 11-0003 (+14.6), 11-0018 (+13.6), 10-0023 (+13.1), 27-0010 (+12.4), 27-0007 (+11.2), 10-0030 (+11.0), 27-0030 (+10.0), 10-0044 (+9.6)

Top losses: 10-0018 (-26.1), 27-0003 (-23.9), 27-0011 (-22.5), 27-0019 (-22.3), 11-0027 (-14.3), 11-0005 (-10.8), 27-0038 (-9.9), 11-0009 (-9.8), 27-0004 (-9.8), 11-0024 (-8.6)
