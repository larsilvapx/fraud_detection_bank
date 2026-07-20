## Distribuição das Classes

<img src="images/distribuicao_classes.png" width="600">

plt.figure(figsize=(6,4))

sns.countplot(data=df, x="Class")

plt.title("Distribuição das Classes")

plt.savefig("../images/distribuicao_classes.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

