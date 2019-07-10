import pandas as pd
import os

def generate_data(review_path, label_path):
    path = '/'.join(review_path.split("/")[:-1])
    review = pd.read_csv(review_path, header=None, sep='\t')
    label = pd.read_csv(label_path, header=None, sep='\t')

    data = pd.merge(review, label, how='inner', right_on=[0, 1, 4], left_on=[0, 1, 2])
    data = data.loc[:,['3_x', '3_y']]

    data = data.rename(index=str, columns={'3_x': 'src', '3_y':'label'})

    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    # split the test/train
    ratio = 0.7
    train_end = int(ratio * len(data))
    train_data = data.iloc[:train_end, :]
    test_data = data.iloc[train_end:, :]
    train_data.to_csv(os.path.join(path, 'train_data.csv'), index=None, header=None)
    test_data.to_csv(os.path.join(path, 'test_data.csv'), index=None, header=None)

    # fake data
    fake_data = train_data[train_data['label'] == -1]
    fake_data.loc[:, 'tgt'] = fake_data['src']
    fake_data.to_csv(os.path.join(path, 'fake_data.csv'), index=None, header=None)

    # real data
    real_data = train_data[train_data['label'] == 1]
    real_data.to_csv(os.path.join(path, 'real_data.csv'), index=None, header=None)

if __name__ == '__main__':
    generate_data("../data/YelpNYC/reviewContent", "../data/YelpNYC/metadata")