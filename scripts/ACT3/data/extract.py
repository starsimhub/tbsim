import pandas as pd

df = pd.read_csv('WPP2024_Life_Table_Complete_Medium_Female_1950-2023.csv') \
    .set_index(['Location', 'Time', 'Sex', 'AgeGrpStart']) \
    .loc['Viet Nam']\
    [['mx']]

dm = pd.read_csv('WPP2024_Life_Table_Complete_Medium_Male_1950-2023.csv') \
    .set_index(['Location', 'Time', 'Sex', 'AgeGrpStart']) \
    .loc['Viet Nam']\
    [['mx']]

df = pd.concat([df, dm], axis=0) \
    .to_csv('Vietnam_ASMR.csv')


# For Births module
df = pd.read_csv('WPP2024_Demographic_Indicators_Medium.csv') \
    .rename(columns={'Time': 'Year'}) \
    .set_index(['Location', 'Year']) \
    .loc['Viet Nam']\
    [['CBR']] \
    .to_csv('Vietnam_CBR.csv')


# For Pregnancy module
df = pd.read_excel('WPP2024_FERT_F01_FERTILITY_RATES_BY_SINGLE_AGE_OF_MOTHER.xlsx', sheet_name='Estimates', skiprows=16) \
    .set_index(['Region, subregion, country or area *', 'Year']) \
    .loc['Viet Nam'] \
    [range(15, 50)] \
    .reset_index() \
    .melt(id_vars=['Year'], var_name='AgeGrp', value_name='ASFR') \
    .rename(columns={'Year': 'Time'}) \
    .set_index(['Time', 'AgeGrp']) \
    .sort_index()

df.to_csv('Vietnam_ASFR.csv')
