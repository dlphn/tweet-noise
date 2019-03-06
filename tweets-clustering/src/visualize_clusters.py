# -*- coding: utf-8 -*-
"""
@author: bmazoyer

edits by dshi, hbaud, vlefranc
"""


import webbrowser
import pandas as pd
import logging
import matplotlib.pyplot as plt

# code adapté de celui des centraliens

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class Visu:

    def __init__(self, data, labels, display="all"):

        self.data = data.assign(pred=pd.Series(labels, dtype=data.label.dtype).values)
        if display == "labeled":
            self.data = self.data[data.label.notnull()]
        self.clusters = self.data.groupby("pred").size().to_frame(name="size")\
            .sort_values("size", ascending=False).reset_index()
        self.html = """
            <!DOCTYPE html>
            <html lang="fr"><head><meta charset="UTF-8">
                <title>Tweet Clusters</title>
                <style>
                    .dropdown {
                        position: relative;
                        display: block;
                    }

                    .dropdown-content {
                        display: none;
                        background-color: #f9f9f9;
                        min-width: 160px;
                        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
                        padding: 12px 16px;
                        z-index: 1;
                    }

                    .dropdown:hover {
                        background-color: lightgrey;
                    }

                    .unselectable {
                        -webkit-user-select: none;
                        -khtml-user-select: none;
                        -moz-user-select: none;
                        -ms-user-select: none;
                        -o-user-select: none;
                        user-select: none;
                        cursor: default;
                    }
                </style>
                <script>
                function toggleList(e) {
                    var l = e.querySelector(".dropdown-content");
                    if (l.style.display == "block") {
                        l.style.display = "none";
                    } else {
                        l.style.display = "block";
                    }
                }
                </script>
                </head>
                <body>\n"""

    # def load_html(self):
    #     for event_id, tweet_ids in self.clusters.items():
    #         self.html += """<div class="dropdown" onClick="toggleList(this)">
    #                     <h4 class="unselectable">Cluster {} (size {}):</h4>\n
    #                     <ul><div class="dropdown-content">\n""".format(event_id, len(self.clusters[event_id]))
    #         for idx in tweet_ids:
    #             text = self.data.loc[self.data["id"] == idx]["text"].iloc[0]
    #             text = text.replace("\n", "\\n")
    #             self.html += "<li>id {}, text: {}</li>\n".format(idx, text)
    #         self.html += """</div></ul></div>\n"""
    #     self.html += """</body></html>\n"""

    def plot(self, path):
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 15))
        df = self.data.groupby(["label", "pred"]).size().unstack().fillna(0)
        head = df.reindex(df.sum().sort_values(ascending=False).index, axis=1).T[:100]
        head.plot(kind="bar",
                  ax=ax1,
                  stacked=True,
                  legend=False
                  )
        middle = df.reindex(df.sum().sort_values(ascending=False).index, axis=1).T[100:200]
        middle.plot(kind="bar",
                    ax=ax2,
                    stacked=True,
                    legend=False
                    )
        middle = df.reindex(df.sum().sort_values(ascending=False).index, axis=1).T[-100:]
        middle.plot(kind="bar",
                    ax=ax3,
                    stacked=True,
                    legend=False
                    )

        plt.savefig("./tmp/barplot_" + path + ".pdf", bbox_inches='tight')
        logging.info("Saved pdf file")

    def compute_proportions(self, row):
        count_spam, count_actualite, count_other = 0, 0, 0
        for _, tweet in self.data[self.data["pred"] == row["pred"]].iterrows():
            if tweet["label"] == 'spam':
                count_spam += 1
            elif tweet["label"] == 'actualité':
                count_actualite += 1
            else:
                count_other += 1
        return count_spam, count_actualite, count_other

    def write_html(self, path):
        for _, row in self.clusters[self.clusters["size"]>2].iterrows():
            nb_spam, nb_actualite, nb_other = self.compute_proportions(row)
            self.html += """<div class="dropdown" onClick="toggleList(this)">
                                <h4 class="unselectable">Cluster {} (size {}): {} spam, {} actualité, {} other</h4>\n
                                <ul><div class="dropdown-content">\n""".format(row["pred"], row["size"], nb_spam, nb_actualite, nb_other)
            for _, tweet in self.data[self.data["pred"] == row["pred"]].iterrows():
                text = tweet["text"].replace("\n", "\\n")
                self.html += "<li>id {}, label {}, text: {}</li>\n".format(tweet["id"], tweet["label"], text)
            self.html += """</div></ul></div>\n"""
        self.html += """</body></html>\n"""

        with open("./tmp/annotated_clusters_" + path + ".html", "w") as f:
            logging.info("Saving HTML file")
            f.write(self.html)

    def open_html(self, path):
        webbrowser.open("./tmp/annotated_clusters_" + path + ".html")

