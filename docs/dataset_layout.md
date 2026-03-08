# Dataset Layout

MobileFetalCLIP expects your licensed datasets in this layout:

```text
data/
  pretraining/
    shards/
      shard_0000000001.tar
      ...
  eval/
    FETAL_PLANES_DB/
      Images/
      FETAL_PLANES_DB_data.csv
    HC18/
      training_set/
      training_set_pixel_size_and_HC.csv
```

## Required files

### FETAL_PLANES_DB CSV
Expected columns:
- `Image_name`
- `Plane`
- `Brain_plane`
- `Train `

### HC18 CSV
Expected columns:
- `filename`
- `pixel size(mm)`
- `head circumference (mm)`

## Validation

Run:

```bash
python tools/validate_dataset.py --strict
```

The validator checks file presence, shard expansion, and the expected CSV columns.
