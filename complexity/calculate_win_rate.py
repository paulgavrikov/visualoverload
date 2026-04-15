import pandas as pd
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)

    df.loc[~df.choice.isin(["A", "B"]), "choice"] = df[~df.choice.isin(["A", "B"])].choice.apply(lambda x: x.split("\n\n")[-1])
    df = df.reset_index()
    
    all_image_names = list(set(df.img_A.unique()).union(set(df.img_B.unique())))
    win_map = np.zeros((len(all_image_names), len(all_image_names)))

    name_to_idx = dict((v, k) for (k, v) in enumerate(all_image_names))

    df["x"] = df.img_A.apply(name_to_idx.get)
    df["y"] = df.img_B.apply(name_to_idx.get)
    df["win"] = (df.choice == "A")

    df.x = df.x.astype(int)
    df.y = df.y.astype(int)

    for idx, row in df.iterrows():
        win_map[row.y, row.x] = 1 if row.win else 0
        win_map[row.x, row.y] = 0 if row.win else 1

    df_win_rates = pd.DataFrame(sorted(list(zip(win_map.mean(0) * (len(all_image_names)/(len(all_image_names)-1)), all_image_names))), 
                                columns=["win_rate", "image"]).set_index("image")
    df_win_rates.to_csv("win_rate.csv")

    print(df_win_rates)