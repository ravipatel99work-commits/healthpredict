# HealthPredict v3 (Educational Demo)

**Disclaimer:** Not for medical use. This is a learning demo only.

## Run
```bash
python -m venv disease_env
# Windows
disease_env\Scripts\activate
# macOS/Linux
source disease_env/bin/activate

pip install -r requirements.txt
python model/merge_datasets.py   # builds merged_dataset.csv (with tiny demo data if missing)
python model/train_model.py      # trains and saves model + features
python app.py
```
Open http://localhost:5000

## Pages
- `/` Landing (hero + CTA)
- `/about` About us
- `/diseases` Grid of 6 diseases (Diabetes, Flu, Pneumonia, Heart, Kidney, Stroke)
- `/predict/<disease>` Disease-specific form
- `/result` Friendly result view
