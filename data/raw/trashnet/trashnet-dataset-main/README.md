# TrashNet Dataset (Resized)

Resized TrashNet dataset used for the AI-powered waste classification proof-of-concept.

## About

The dataset contains photographs of household waste items, organized into six
classes. It is a resized (512 × 384) version of the original
[TrashNet](https://github.com/garythung/trashnet) dataset by Thung & Yang
(Stanford CS 229).

- **Source:** https://github.com/garythung/trashnet
- **Citation:**
  Thung, Gary and Yang, Mindy. *Classification of Trash for Recyclability Status.*
  CS229 Project Report, Stanford University, 2016.
  https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf

## Contents

- **Classes:** `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- **Total images:** 2527
- **Format:** JPEG
- **Dimensions:** 512 × 384

## Class distribution

| Class     | Images | %    |
|-----------|--------|------|
| cardboard | 403    | 15.9 |
| glass     | 501    | 19.8 |
| metal     | 410    | 16.2 |
| paper     | 594    | 23.5 |
| plastic   | 482    | 19.1 |
| trash     | 137    | 5.4  |
| **Total** | **2527** | **100** |

## Folder structure

```
dataset-resized/
├── cardboard/  (403 images)
├── glass/      (501 images)
├── metal/      (410 images)
├── paper/      (594 images)
├── plastic/    (482 images)
└── trash/      (137 images)
```
