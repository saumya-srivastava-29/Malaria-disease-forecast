import pandas as pd

def get_clean_data(filepath):
    df = pd.read_csv(filepath, skiprows=4)

    # Drop unnecessary columns
    df = df.drop(columns=["Country Code", "Indicator Name", "Indicator Code"])

    # Convert from wide to long format
    df_long = df.melt(id_vars="Country Name", var_name="year", value_name="incidence")

    # Drop missing values
    df_long = df_long.dropna()

    # Convert year column to datetime
    df_long["year"] = pd.to_datetime(df_long["year"], format="%Y")

    # Rename columns
    df_long = df_long.rename(columns={"Country Name": "country"})

    return df_long
