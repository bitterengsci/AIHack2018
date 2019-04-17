import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow

from sklearn.decomposition import PCA

if __name__ == "__main__":
    # https://harrisonzhu508.github.io/Documentation/california.html
    pd.set_option('display.max_columns', None)

    ''' ##############
    Read .csv file
    ############## '''

    data_dir = "~/PycharmProjects/hack2018/california/train/"
    label_df = pd.read_csv(data_dir + "BG_METADATA_2016.csv")
    columnnames = {}
    for row in label_df.iterrows():
        columnnames[row[1][1]] = row[1][2]

    ''' ##############
        Load all files
        ############## '''

    df1 = pd.read_csv(data_dir + "X01_AGE_AND_SEX.csv")                 # 1
    df2 = pd.read_csv(data_dir + "X02_RACE.csv")                        # 2
    df3 = pd.read_csv(data_dir + "X03_HISPANIC_OR_LATINO_ORIGIN.csv")   # 3
    df7 = pd.read_csv(data_dir + "X07_MIGRATION.csv")                   # 4
    df8 = pd.read_csv(data_dir + "X08_COMMUTING.csv")
    df9 = pd.read_csv(data_dir + "X09_CHILDREN_HOUSEHOLD_RELATIONSHIP.csv")     # 5
    df11 = pd.read_csv(data_dir + "X11_HOUSEHOLD_FAMILY_SUBFAMILIES.csv")       # 6
    df12 = pd.read_csv(data_dir + "X12_MARITAL_STATUS_AND_HISTORY.csv")         # 7
    df14 = pd.read_csv(data_dir + "X14_SCHOOL_ENROLLMENT.csv")                  # 8
    df15 = pd.read_csv(data_dir + "X15_EDUCATIONAL_ATTAINMENT.csv")             # 9
    df16 = pd.read_csv(data_dir + "X16_LANGUAGE_SPOKEN_AT_HOME.csv")            # 10
    df17 = pd.read_csv(data_dir + "X17_POVERTY.csv")                            # 11
    df19 = pd.read_csv(data_dir + "X19_INCOME.csv")                             # 12
    df20 = pd.read_csv(data_dir + "X20_EARNINGS.csv")                           # 13
    df21 = pd.read_csv(data_dir + "X21_VETERAN_STATUS.csv")
    df22 = pd.read_csv(data_dir + "X22_FOOD_STAMPS.csv")                        # 14
    df23 = pd.read_csv(data_dir + "X23_EMPLOYMENT_STATUS.csv")                  # 15
    df27 = pd.read_csv(data_dir + "X27_HEALTH_INSURANCE.csv")                   # 16

    # df8 df21 没用

    # lll = [1: df1, 2: df2, 3: df3, 4: df7, 5: df9,
    #         6: df11, 7:df12, 8:df14, 9:df15, 10:df16,
    #         11:df17, 12:df19, 13:df20, 14:df22, 15:df23, 16: df27]

    list_dataset = [df1, df2, df3, df7, df9, df11, df12, df14, df15, df16, df17, df19, df20, df22, df23, df27]

    correlated_features = list()
    result_name = "Correlation_0.7"
    file_object = open(result_name, 'a')

    iteration = 0
    useful_data = dict()

    for i in list_dataset:

        searching_file = i

        searching_file.rename(columns=columnnames, inplace=True)
        searching_file = searching_file.drop(columns="GEOID")  # drop the GEOID column

        df20 = pd.read_csv(data_dir + "X20_EARNINGS.csv")
        df20.rename(columns=columnnames, inplace=True)

        interesting_feature = "SEX BY EARNINGS IN THE PAST 12 MONTHS (IN 2016 INFLATION-ADJUSTED DOLLARS) FOR THE POPULATION 16 YEARS AND OVER WITH EARNINGS IN THE PAST 12 MONTHS: Male: $100,000 or more: Population 16 years and over with earnings -- (Estimate)"

        for item in list(searching_file):
            correlations = searching_file[item].corr(df20[interesting_feature], method='pearson')
            if correlations > 0.6 and (not correlations == 1):
                correlated_features.append(str(item))           # headers
                # print(str(item))
                # print(" ")
                raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                            'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                            'age': [42, 52, 36, 24, 73],
                            'preTestScore': [4, 24, 31, 2, 3],
                            'postTestScore': [25, 94, 57, 62, 70]}

                useful_data[str(item)] = searching_file[item]

                file_object.write(str(item))
                file_object.write("\n")
        iteration += 1
        print(iteration, "number of correlated features ", len(correlated_features))

    print("number of correlated features ", len(correlated_features))
    output_csv = pd.DataFrame(useful_data, columns=correlated_features)
    output_csv.to_csv('output.csv')
    file_object.close()
