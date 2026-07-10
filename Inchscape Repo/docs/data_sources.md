# Data sources

## Raw sales extract

`data/raw/chile_suzuki_historical_sales.csv` is the supplied historical sales extract. It is excluded from Git because it is approximately 330 MB, which is larger than GitHub's normal file-size limit. Place the supplied file at that path before running the preparation script.

## External features

`data/external/monthly_external_features.csv` is the supplied monthly weather, calendar, and road-safety feature set. It is retained separately from the sales data so it can be joined by month during modelling.

## Data dictionary

`forecasting_data_dictionary.xlsx` is the supplied source dictionary. It remains in this folder unchanged for reference.

## Processed sales outputs

`src/prepare_data.py` creates the processed files from the raw extract. Each contains only the identifiers, monthly demand, demand-type label, collision flag, and available location/channel context needed for analysis. This keeps the working data compact while preserving the original source in `data/raw/`.
