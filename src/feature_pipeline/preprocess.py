# Preprocessing the raw splitted train, eval and holdout datasets by performing the following tasks : 
# - Add 'lat' and 'lng' from the metro dataset
# - Remove the duplicate row of the dataset while ignoring the 'date' and 'year' column
# - Remove outliers (too high prices)

# Before executing this script, make sure that the original raw data was already splitted. If this is not the case, it can be done by running the script 'load_and_split.py'.

import pandas as pd
from pathlib import Path

INPUT_DATA_DIR = Path('data/raw/')
OUTPUT_DATA_DIR = Path('data/processed/')
USMETRO_DATASET_FILEPATH = Path('data/raw/usmetros.csv')
# Mapping from the cities present in the housing price dataset (left) and the cities present in the usmetro dataset (right)
CITY_MAPPING = { 
    'Las Vegas-Henderson-Paradise': 'Las Vegas-Henderson-North Las Vegas',
    'Denver-Aurora-Lakewood': 'Denver-Aurora-Centennial',
    'Houston-The Woodlands-Sugar Land': 'Houston-Pasadena-The Woodlands',
    'Austin-Round Rock-Georgetown': 'Austin-Round Rock-San Marcos',
    'Miami-Fort Lauderdale-Pompano Beach': 'Miami-Fort Lauderdale-West Palm Beach',
    'San Francisco-Oakland-Berkeley': 'San Francisco-Oakland-Fremont',
    'DC_Metro': 'Washington-Arlington-Alexandria',
    'Atlanta-Sandy Springs-Alpharetta': 'Atlanta-Sandy Springs-Roswell'
}
PRICE_OUTLIER_CUTOFF = 19_000_000 # $

def add_city_coordinates(
                        df: pd.DataFrame,
                        usmetro_dataset: Path | str = USMETRO_DATASET_FILEPATH,
) -> pd.DataFrame :
    # Adds the city coordinates (lat and lng) to the df DataFrame by merging it with the usmetro dataset if available
    # Returns the DataFrame with the additional 'lat' and 'lng' columns

    # Skip is usmetre dataset not available
    if usmetro_dataset==None:
        print('⚠️  USMetro dataset filepath no given : lat/lng were not added')
        return df

    # Load USMetro Dataset (source : https://simplemaps.com/data/us-metros (Basic))
    df_metro = pd.read_csv(usmetro_dataset)
    df_metro['metro_full'] = df_metro['metro_full'].str.split(',').str[0]

    # Merge 'lat' and 'lng'
    df["city_full"] = df["city_full"].replace(CITY_MAPPING)
    df = pd.merge(left=df, right=df_metro[['metro_full', 'lat', 'lng']], left_on='city_full', right_on='metro_full', how='left')

    # Check if 'lat' and 'lng' are missing for some rows
    missing = df[df['lat'].isnull()]
    if len(missing)==0:
        print('✅ All cities matched with metro dataset')
    else :
        print('❌ ' + str(len(missing)) + ' rows missing')
    
    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame :
    # Removes the duplicate row of the dataset while ignore the 'date' and 'year' column

    number_duplicate_row_before = df[df.duplicated(subset=df.columns.difference(['date', 'year']))].shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(['date', 'year']), keep=False)
    number_duplicate_row_after = df[df.duplicated(subset=df.columns.difference(['date', 'year']))].shape[0]

    if number_duplicate_row_after == 0:
        print(f'✅ All duplicates were successfully removed ({number_duplicate_row_before} were present)')
    else :
        print(f'❌ {number_duplicate_row_after} duplicates still present')

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame :
    # Remove outliers and returns the resulting DataFrame 

    number_outliers_before = df[df['median_list_price'] > PRICE_OUTLIER_CUTOFF].shape[0]
    df = df[df['median_list_price'] <= PRICE_OUTLIER_CUTOFF].copy()
    number_outliers_after = df[df['median_list_price'] > PRICE_OUTLIER_CUTOFF].shape[0]

    if number_outliers_after == 0:
        print(f'✅ All outliers were removed ({number_outliers_before} were present)')
    else :
        print(f'❌ {number_outliers_after} outliers still present')

    return df


def save_cleaned_data(
                df: pd.DataFrame | str,
                filename : str,
                output_dir: Path | str = OUTPUT_DATA_DIR        
) -> pd.DataFrame :
    # Save the DataFrame df in the directory output_dir under the name filename as .csv file
    # Returns the save DataFrame

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / filename, index=False)
    print(f'✅ {output_dir / filename} saved')

    return df


def preprocess_split(
                split: str,
                input_dir: Path | str = INPUT_DATA_DIR,
                output_dir: Path | str = OUTPUT_DATA_DIR,
                usmetro_dataset: Path | str = USMETRO_DATASET_FILEPATH,
) -> pd.DataFrame :
    # Preprocess one split (train, eval or holdout) and save the preprocessed datasets as .csv file

    print(f'Process split {split} :')
    df = pd.read_csv(input_dir / split)
    df = add_city_coordinates(df, usmetro_dataset) 
    df = drop_duplicates(df)
    df = remove_outliers(df)
    save_cleaned_data(df, f'preprocessed_{split}', output_dir)

    return df



def preprocess(
                splits : tuple[str] = ('train.csv', 'eval.csv', 'holdout.csv'),
                input_dir: Path | str = INPUT_DATA_DIR,
                output_dir: Path | str = OUTPUT_DATA_DIR,
                usmetro_dataset: Path | str = USMETRO_DATASET_FILEPATH,
):
    # Preprocess several splits of datasets and save it as .csv in the output_dir

    print('Preprocessing : ')
    for split in splits:
        preprocess_split(split, input_dir, output_dir, usmetro_dataset)
    print('')


if __name__ == '__main__':
    preprocess()

