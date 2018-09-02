import pandas as pd

def process_raw_data(file_in, file_out):

    df = pd.read_csv(file_in)
    # print(df['blood_sugar'])
    # print(df['the_amount_of_glucose_in_a_persons_blood'])
    # print(df['is_it_compositional'])
    data_table = df.iloc[:,[14,16,18]]
    # print(data_table)

    data_table = data_table.groupby(['blood_sugar','the_amount_of_glucose_in_a_persons_blood']).mean()
    data_table.to_csv(file_out)

file_name = 'preprocessed_data.csv'
df = pd.read_csv(file_name, names = ['phrase','scenario','cd_score'])
# print(df)
print(df['phrase'].values)
# for index, row in df.iterrows():
#    print(row['phrase'], row['scenario'])
