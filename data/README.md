# Apple Dataset

This directory contains organized apple images for training the AI models.

## Directory Structure:

```
data/
├── varieties/          # Apple variety classification
│   ├── Sharbati/
│   ├── Sunehari/
│   ├── Maharaji/
│   ├── Splendour/
│   ├── Himsona/
│   ├── Himkiran/
├── health/             # Health classification
│   ├── healthy/        # Fresh apples
│   └── rotten/         # Spoiled apples
└── surface/            # Surface treatment
    ├── waxed/          # Wax-treated apples
    └── unwaxed/        # Natural apples
```

## Next Steps:
1. Add your apple images to the appropriate folders
2. Run data validation: `python scripts/data_validator.py`
3. Start training: `python scripts/train_all_models.py`
