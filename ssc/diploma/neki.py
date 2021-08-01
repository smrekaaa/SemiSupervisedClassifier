import pandas as pd

if __name__ == '__main__':

    # Settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Navadni
    s1 = [
        ['t1', 'm1', 'KNN', 0.6, 0.3],
        ['t1', 'm1', 'CART', 0.7, 0.2],
        ['t1', 'm1', 'NB', 0.8, 0.38]
    ]
    df1 = pd.DataFrame(s1, columns=['Type', 'Data', 'Classifier', 'accuracy', 'f1_score'])

    s2 = [
        ['t1', 'm2', 'KNN', 0.2, 0.1],
        ['t1', 'm2', 'CART', 0.5, 0.1],
        ['t1', 'm2', 'NB', 0.8, 0.34]
    ]
    df2 = pd.DataFrame(s2, columns=['Type', 'Data', 'Classifier', 'accuracy', 'f1_score'])

    s3 = [
        ['t1', 'm3', 'KNN', 0.1, 0.54],
        ['t1', 'm3', 'CART', 0.4, 0.87],
        ['t1', 'm3', 'NB', 0.9, 0.8]
    ]
    df3 = pd.DataFrame(s3, columns=['Type', 'Data', 'Classifier', 'accuracy', 'f1_score'])

    # SSC
    ss1 = [
        ['t2', 'm1', 'KNN', 0.6, 0.3],
        ['t2', 'm1', 'CART', 0.7, 0.2],
        ['t2', 'm1', 'NB', 0.8, 0.38]
    ]
    df1s = pd.DataFrame(ss1, columns=['Type', 'Data', 'Classifier', 'accuracy', 'f1_score'])

    ss2 = [
        ['t2', 'm2', 'KNN', 0.2, 0.1],
        ['t2', 'm2', 'CART', 0.5, 0.1],
        ['t2', 'm2', 'NB', 0.8, 0.34]
    ]
    df2s = pd.DataFrame(ss2, columns=['Type', 'Data', 'Classifier', 'accuracy', 'f1_score'])

    ss3 = [
        ['t2', 'm3', 'KNN', 0.1, 0.54],
        ['t2', 'm3', 'CART', 0.4, 0.87],
        ['t2', 'm3', 'NB', 0.9, 0.8]
    ]
    df3s = pd.DataFrame(ss3, columns=['Type', 'Data', 'Classifier', 'accuracy', 'f1_score'])

    # print(df1)
    # print()
    # print(df2)
    # print()
    # print(df3)
    #
    # print(df1s)
    # print()
    # print(df2s)
    # print()
    # print(df3s)

    dfc = pd.concat([df1, df2, df3, df1s, df2s, df3s])
    dfc.sort_values(['Classifier', 'Data', 'Type'], inplace=True)
    dfc.reset_index(drop=True, inplace=True)
    #print(dfc)
