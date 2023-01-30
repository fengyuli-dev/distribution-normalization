import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="darkgrid")
sns.set(rc={"savefig.dpi":800})

dataset = "mscoco"
accuracy = "Top10"
t2i_path = f"retrieval_ablation_data/t2i_{dataset}.csv"
i2t_path = f"retrieval_ablation_data/i2t_{dataset}.csv"

# t2i_path = "retrieval_ablation_data/yifei_t2i.csv"
# i2t_path = "retrieval_ablation_data/yifei_i2t.csv"
t2i_df = pd.read_csv(t2i_path)[:50]
for col in ["CLIP Top1","CLIP Top5","CLIP Top10","CLIP-DN Top1","CLIP-DN Top5",
            "CLIP-DN Top10", "CLIP Top1 STD", "CLIP Top5 STD", "CLIP Top10 STD",
            "CLIP-DN Top1 STD", "CLIP-DN Top5 STD","CLIP-DN Top10 STD"]:
    t2i_df[col] = t2i_df[col].apply(lambda x: x*100)
i2t_df = pd.read_csv(i2t_path)[:50]

# add in zeroshot results
t2i_zeroshot = pd.DataFrame({"Epochs Finetuned": 0, 
            "CLIP Top1": 30.2, "CLIP Top5": 55.1, "CLIP Top10": 66.4, 
            "CLIP-DN Top1": 32.1, "CLIP-DN Top5": 57.4, "CLIP-DN Top10": 68.3, 
            "CLIP Top1 STD": 0., "CLIP Top5 STD": 0., "CLIP Top10 STD": 0.,
            "CLIP-DN Top1 STD": 0., "CLIP-DN Top5 STD": 0.,"CLIP-DN Top10 STD": 0.}, index=[0])
t2i_df = pd.concat((t2i_zeroshot, t2i_df.loc[:])).reset_index(drop=True)

i2t_zeroshot = pd.DataFrame({"Epochs Finetuned": 0, 
            "CLIP Top1": 52.4, "CLIP Top5": 76.0, "CLIP Top10": 84.5, 
            "CLIP-DN Top1": 52.7, "CLIP-DN Top5": 76.4, "CLIP-DN Top10": 84.8, 
            "CLIP Top1 STD": 0., "CLIP Top5 STD": 0., "CLIP Top10 STD": 0.,
            "CLIP-DN Top1 STD": 0., "CLIP-DN Top5 STD": 0.,"CLIP-DN Top10 STD": 0.}, index=[0])
i2t_df = pd.concat((i2t_zeroshot, i2t_df.loc[:])).reset_index(drop=True)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(10, 5))

target_df = t2i_df

# First plot
df = pd.DataFrame(t2i_df, columns=["Epochs Finetuned", f"CLIP {accuracy}", f"CLIP-DN {accuracy}"])
retrieval_plot = sns.lineplot(x="Epochs Finetuned", y="value", hue="variable", 
                                data=pd.melt(df, id_vars=["Epochs Finetuned"], 
                                value_vars=[f"CLIP {accuracy}", f"CLIP-DN {accuracy}"]), ax=ax1)
retrieval_plot.set(ylabel='Accuracy')
retrieval_plot.set_title(f"T2I: Accuracy @ {accuracy}")
retrieval_plot.legend(title="Model Type", loc="upper right")
legend_handles, _= retrieval_plot.get_legend_handles_labels()
new_labels=[f"CLIP {accuracy}", f"CLIP+DN {accuracy}"]
retrieval_plot.legend(legend_handles, new_labels)

# calculate standard deviation for each line
clip_std = t2i_df[f"CLIP {accuracy} STD"]
clip_dn_std = t2i_df[f"CLIP-DN {accuracy} STD"]

# create x and y values for the +/- standard deviation lines
x = t2i_df["Epochs Finetuned"]
y1 = t2i_df[f"CLIP {accuracy}"] + clip_std
y2 = t2i_df[f"CLIP {accuracy}"] - clip_std
y3 = t2i_df[f"CLIP-DN {accuracy}"] + clip_dn_std
y4 = t2i_df[f"CLIP-DN {accuracy}"] - clip_dn_std

# fill area between the line and the +/- standard deviation lines
clip_color = retrieval_plot.lines[0].get_color()
clip_dn_color = retrieval_plot.lines[1].get_color()
ax1.fill_between(x, y1, y2, color=clip_color, alpha=0.2)
ax1.fill_between(x, y3, y4, color=clip_dn_color, alpha=0.2)

# Second plot
df = pd.DataFrame(i2t_df, columns=["Epochs Finetuned", f"CLIP {accuracy}", f"CLIP-DN {accuracy}"])
retrieval_plot = sns.lineplot(x="Epochs Finetuned", y="value", hue="variable", 
                                data=pd.melt(df, id_vars=["Epochs Finetuned"], 
                                value_vars=[f"CLIP {accuracy}", f"CLIP-DN {accuracy}"]), ax=ax2)

retrieval_plot.set(ylabel='Accuracy')
retrieval_plot.set_title(f"I2T: Accuracy @ {accuracy}")
retrieval_plot.legend(title="Model Type", loc="upper right")
legend_handles, _= retrieval_plot.get_legend_handles_labels()
new_labels=[f"CLIP {accuracy}", f"CLIP+DN {accuracy}"]
retrieval_plot.legend(legend_handles, new_labels)   

# calculate standard deviation for each line
clip_std = i2t_df[f"CLIP {accuracy} STD"]
clip_dn_std = i2t_df[f"CLIP-DN {accuracy} STD"]

# create x and y values for the +/- standard deviation lines
x = i2t_df["Epochs Finetuned"]
y1 = i2t_df[f"CLIP {accuracy}"] + clip_std
y2 = i2t_df[f"CLIP {accuracy}"] - clip_std
y3 = i2t_df[f"CLIP-DN {accuracy}"] + clip_dn_std
y4 = i2t_df[f"CLIP-DN {accuracy}"] - clip_dn_std

# fill area between the line and the +/- standard deviation lines
clip_color = retrieval_plot.lines[0].get_color()
clip_dn_color = retrieval_plot.lines[1].get_color()
ax2.fill_between(x, y1, y2, color=clip_color, alpha=0.2)
ax2.fill_between(x, y3, y4, color=clip_dn_color, alpha=0.2)

plt.savefig(f"retrieval_ablation_data/retrieval_ablations_{accuracy}.png") 


# average retrieval accuracy
print(f"For {accuracy=}")
clip_avg_acc_t2i = np.mean(t2i_df[f"CLIP {accuracy}"])
clip_dn_avg_acc_t2i = np.mean(t2i_df[f"CLIP-DN {accuracy}"])
clip_avg_acc_i2t = np.mean(i2t_df[f"CLIP {accuracy}"])
clip_dn_avg_acc_i2t = np.mean(i2t_df[f"CLIP-DN {accuracy}"])
print(f"{clip_avg_acc_t2i=:.2f}, {clip_dn_avg_acc_t2i=:.2f}, {clip_avg_acc_i2t=:.2f}, {clip_dn_avg_acc_i2t=:.2f}")

# average retrieval std
clip_avg_std_t2i = np.mean(t2i_df[f"CLIP {accuracy} STD"])
clip_dn_avg_std_t2i = np.mean(t2i_df[f"CLIP-DN {accuracy} STD"])
clip_avg_std_i2t = np.mean(i2t_df[f"CLIP {accuracy} STD"])
clip_dn_avg_std_i2t = np.mean(i2t_df[f"CLIP-DN {accuracy} STD"])
print(f"{clip_avg_std_t2i=:.2f}, {clip_dn_avg_std_t2i=:.2f}, {clip_avg_std_i2t=:.2f}, {clip_dn_avg_std_i2t=:.2f}")


# largest difference in accuracy
clip_high_acc_t2i = np.max(t2i_df[f"CLIP {accuracy}"])
clip_dn_high_acc_t2i = np.max(t2i_df[f"CLIP-DN {accuracy}"])
clip_high_acc_i2t = np.max(i2t_df[f"CLIP {accuracy}"])
clip_dn_high_acc_i2t = np.max(i2t_df[f"CLIP-DN {accuracy}"])
clip_low_acc_t2i = np.min(t2i_df[f"CLIP {accuracy}"])
clip_dn_low_acc_t2i = np.min(t2i_df[f"CLIP-DN {accuracy}"])
clip_low_acc_i2t = np.min(i2t_df[f"CLIP {accuracy}"])
clip_dn_low_acc_i2t = np.min(i2t_df[f"CLIP-DN {accuracy}"])
print(f"{clip_high_acc_t2i=:.2f}, {clip_low_acc_t2i=:.2f}, {clip_dn_high_acc_t2i=:.2f}, {clip_dn_low_acc_t2i=:.2f}")
print(f"{clip_high_acc_t2i - clip_low_acc_t2i=:.2f}, {clip_dn_high_acc_t2i - clip_dn_low_acc_t2i=:.2f}")
print(f"{clip_high_acc_t2i - clip_dn_high_acc_t2i=:.2f}")
print(f"{clip_high_acc_i2t=:.2f}, {clip_low_acc_i2t=:.2f}, {clip_dn_high_acc_i2t=:.2f}, {clip_dn_low_acc_i2t=:.2f}")
print(f"{clip_high_acc_i2t - clip_low_acc_i2t=:.2f}, {clip_dn_high_acc_i2t - clip_dn_low_acc_i2t=:.2f}")
print(f"{clip_high_acc_i2t - clip_dn_high_acc_i2t=:.2f}")


t2i_clip_initial_improvement = t2i_df[f"CLIP {accuracy}"][1] - t2i_df[f"CLIP {accuracy}"][0]
print(f"{t2i_clip_initial_improvement=:.2f}")
t2i_clip_dn_initial_improvement = t2i_df[f"CLIP-DN {accuracy}"][1] - t2i_df[f"CLIP-DN {accuracy}"][0]
print(f"{t2i_clip_dn_initial_improvement=:.2f}")
i2t_clip_initial_improvement = i2t_df[f"CLIP {accuracy}"][1] - i2t_df[f"CLIP {accuracy}"][0]
print(f"{i2t_clip_initial_improvement=:.2f}")
i2t_clip_dn_initial_improvement = i2t_df[f"CLIP-DN {accuracy}"][1] - i2t_df[f"CLIP-DN {accuracy}"][0]
print(f"{i2t_clip_dn_initial_improvement=:.2f}")


t2i_better_avg = np.mean(t2i_df[f"CLIP-DN {accuracy}"] - t2i_df[f"CLIP {accuracy}"])
print(f"{t2i_better_avg=:.2f}")
i2t_better_avg = np.mean(i2t_df[f"CLIP-DN {accuracy}"] - i2t_df[f"CLIP {accuracy}"])
print(f"{i2t_better_avg=:.2f}")

t2i_better_avg = np.mean(t2i_df[f"CLIP-DN {accuracy}"][:10] - t2i_df[f"CLIP {accuracy}"][:10])
i2t_better_avg = np.mean(i2t_df[f"CLIP-DN {accuracy}"][:10] - i2t_df[f"CLIP {accuracy}"][:10])
avg_diff_10 = (t2i_better_avg + i2t_better_avg) / 2
t2i_better_avg = np.mean(t2i_df[f"CLIP-DN {accuracy}"][10:30] - t2i_df[f"CLIP {accuracy}"][10:30])
i2t_better_avg = np.mean(i2t_df[f"CLIP-DN {accuracy}"][10:30] - i2t_df[f"CLIP {accuracy}"][10:30])
avg_diff_30 = (t2i_better_avg + i2t_better_avg) / 2
t2i_better_avg = np.mean(t2i_df[f"CLIP-DN {accuracy}"][30:50] - t2i_df[f"CLIP {accuracy}"][30:50])
i2t_better_avg = np.mean(i2t_df[f"CLIP-DN {accuracy}"][30:50] - i2t_df[f"CLIP {accuracy}"][30:50])
avg_diff_50 = (t2i_better_avg + i2t_better_avg) / 2
print(f"{avg_diff_10=:.2f}")
print(f"{avg_diff_30=:.2f}")
print(f"{avg_diff_50=:.2f}")

std_dev_avg = np.mean(i2t_df[f"CLIP-DN {accuracy} STD"] + i2t_df[f"CLIP {accuracy} STD"])
print(f"{std_dev_avg=:.2f}")