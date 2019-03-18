# -*- coding: utf-8 -*-
"""
@author: bmazoyer

edits by dshi, hbaud, vlefranc
"""


import webbrowser
import pandas as pd
import logging
import matplotlib.pyplot as plt
from src.eval import compute_proportions

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

    def plot(self, path, label="label"):
        if label == 'category':
            categories = ['actualité', 'reaction', 'conversation', 'publicité', 'bot', 'other spam']
            colors = ['#368bf9', '#9dc7f9', '#fc7e00', '#ffbf00', '#ffe900', '#f74036']
        else:
            categories = ['actualité', 'spam']
            colors = ['#368bf9', '#fc7e00']
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 15))
        df = self.data.groupby([label, "pred"]).size().unstack().fillna(0)
        data = df.reindex(df.sum().sort_values(ascending=False).index, axis=1)
        data = data.reindex(categories)
        head = data.T[1:100]  # Don't display cluster -1
        head.plot(kind="bar",
                  ax=ax1,
                  stacked=True,
                  legend=True,
                  color=colors
                  )
        middle = data.T[100:200]
        middle.plot(kind="bar",
                    ax=ax2,
                    stacked=True,
                    legend=False,
                    color=colors
                    )
        middle = data.T[-100:]
        middle.plot(kind="bar",
                    ax=ax3,
                    stacked=True,
                    legend=False,
                    color=colors
                    )

        plt.savefig("./tmp/barplot_" + label + "_" + path + ".pdf", bbox_inches='tight')
        logging.info("Saved pdf file")

    def write_html(self, path):
        for _, row in self.clusters[self.clusters["size"] > 2].iterrows():
            nb_spam, nb_actualite, nb_other = compute_proportions(self.data, row)
            self.html += """<div class="dropdown" onClick="toggleList(this)">
                                <h4 class="unselectable">Cluster {} (size {}): {} spam, {} actualité, {} other</h4>\n
                                <ul><div class="dropdown-content">\n""".format(row["pred"], row["size"], nb_spam, nb_actualite, nb_other)
            for _, tweet in self.data[self.data["pred"] == row["pred"]].iterrows():
                text = tweet["text"].replace("\n", "\\n")
                self.html += "<li>id {}, label {}, category {}, text: {}</li>\n".format(tweet["id"], tweet["label"], tweet["category"], text)
            self.html += """</div></ul></div>\n"""
        self.html += """</body></html>\n"""

        with open("./tmp/annotated_clusters_" + path + ".html", "w") as f:
            logging.info("Saving HTML file")
            f.write(self.html)

    def open_html(self, path):
        webbrowser.open("./tmp/annotated_clusters_" + path + ".html")

