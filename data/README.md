# Data files (not included)

The public repository intentionally does not contain the original respondent-level WVS Wave 7 dataset or the processed train/test CSV files. Obtain the WVS Wave 7 data through the official World Values Survey data portal and follow its current access and redistribution terms. Do not commit data unless you have confirmed that public redistribution is permitted.

For the original analysis notebooks, place the downloaded Wave 7 CSV at:

`data/WVS_Cross-National_Wave_7_csv_v6_0.csv`

The processed dataset archive used during this project was `processed.zip`, containing `WVS_Train_Clean.csv`, `WVS_Test_Clean.csv`, and `WVS_Feature_Metadata.csv`. Keep these local or use an approved private storage location. The app itself does not require the respondent-level data; it uses the fitted model bundle in `models/`.
