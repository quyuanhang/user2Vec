import sys
import pandas as pd

if __name__ == '__main__':
    df = pd.read_table(sys.argv[1], header=None)
    df.columns = ['boss', 'geek']

    train_frame = pd.DataFrame()
    test_frame = pd.DataFrame()
    neg_frame = pd.DataFrame()

    boss_act_n = df['boss'].value_counts().drop_duplicates()
    boss_act_n = dict(boss_act_n[boss_act_n > 5])
    print(len(boss_act_n))
    df.index = df['boss']
    # df['value_counts'] = df['boss'].map(lambda x: boss_act_n.get(x, 0))

    for boss in boss_act_n:
        boss_action = df.loc[boss, ['boss', 'geek']]
        train_frame = train_frame.append(boss_action.iloc[:-3])
        test_frame = test_frame.append(boss_action.iloc[-3:])
    print(len(train_frame), len(test_frame))
    all_geek = set(train_frame['geek'])

    for boss in boss_act_n:
        boss_action = df.loc[boss, ['boss', 'geek']]
        negative = list(all_geek - set(boss_action['geek'].values))
        negative_sample = pd.Series(negative).sample(n=100)
        negative_sample.index = range(100)
        negative_sample.name = boss
        neg_frame = neg_frame.append(negative_sample)

    # pd.DataFrame(neg_list).to_csv('Data/interview.test.negative', index=False, header=False)

    boss_dict = {k:v for v, k in enumerate(train_frame['boss'].drop_duplicates())}
    geek_dict = {k:v for v, k in enumerate(train_frame['geek'].drop_duplicates())}
    print(len(boss_dict), len(geek_dict))

    train_frame['boss'] = train_frame['boss'].map(lambda x: boss_dict[x])
    train_frame['geek'] = train_frame['geek'].map(lambda x: geek_dict[x])

    test_frame = pd.DataFrame([
        [boss_dict[boss], geek_dict[geek]] 
        for boss, geek in test_frame.values 
        if boss in boss_dict and geek in geek_dict],
        columns=['boss', 'geek'])
    test_frame.index = test_frame['boss']
    test_frame = test_frame.drop_duplicates(subset='boss')
    neg_frame.index = [boss_dict[x] for x in neg_frame.index]
    neg_frame = neg_frame.astype(int)
    # neg_frame = neg_frame.applymap(lambda x: x+1)
    neg_frame = neg_frame.applymap(lambda x: geek_dict.get(x, None))
    neg_frame = pd.concat([test_frame, neg_frame], axis=1, join='inner')
    neg_frame = neg_frame.drop(columns=['boss', 'geek'])

    print(len(train_frame), len(test_frame), len(neg_frame))

    train_frame.to_csv('Data/interview.train', index=False, header=False, sep='\t')
    neg_frame.to_csv('Data/interview.test', index=False, header=False)

    
    
        


