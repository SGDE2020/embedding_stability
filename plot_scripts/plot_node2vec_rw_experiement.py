import matplotlib.pyplot as plt
import csv
import math

walk_length = []
overlap_score = []
cosine_similarity = []
angle = []
with open("results_walk_length_experiment.csv") as f:
    reader = csv.reader(f, delimiter=" ")

    for line in reader:
        if line[0] == '#':
            continue
        print(line)
        walk_length.append(float(line[0]))
        overlap_score.append(float(line[1]))
        cosine_similarity.append(float(line[2]))
        angle.append(math.degrees(math.acos(float(line[2]))))

fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(111)
ax1.set_ylabel("Jaccard Similarity")
ax1.set_ylim(bottom=.69, top=.83)
# ax1.set_title("Stability vs. Walk Length")
ln1 = ax1.plot(walk_length, overlap_score, marker="o", label="Jaccard Similarity", color="#00549F")
ax2 = ax1.twinx()
ln2 = ax2.plot(walk_length, cosine_similarity, marker="o", color="#E30066", label="Cosine Similarity")
ax2.set_ylabel("Cosine Similarty")
ax2.set_ylim(bottom=.93, top=1)
# ax2.set_xlabel("Length of Random Walks")
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=4)
ax1.set_xlabel("Length of Random Walks")
fig.set_size_inches(6, 2.5)
fig.tight_layout()
fig.savefig("plot.pdf")
