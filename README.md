<h1 align="left">
EKG aritmijų klasifikavimas
</h1>

<p align="left">
Mašininio mokymosi projektas EKG signalų analizei ir aritmijų aptikimui naudojant MIT-BIH Arrhythmia Database duomenų rinkinį.
</p>

---

# Projekto aprašymas

Šiame projekte sukurtas klasifikatorius, kuris automatiškai skirsto EKG širdies dūžius į dvi klases:

- normalius dūžius,
- aritminius dūžius.

Modeliui kurti naudoti šie metodai:

- Logistic Regression,
- Random Forest,
- Soft Voting Ensemble.

Klasių disbalanso problemai mažinti pritaikytas SMOTE metodas.

---

# Darbo tikslas

Taikant signalų apdorojimo ir mašininio mokymosi metodus sukurti modelį, galintį klasifikuoti elektrokardiogramos signalus ir nustatyti, ar širdies ritmas atitinka nustatytas normas.

---

# Duomenų rinkinys

Naudotas **MIT-BIH Arrhythmia Database** duomenų rinkinys.

Naudoti failų tipai:

```text
project/
│
├── dataset/
│   ├── 100.dat
│   ├── 100.hea
│   ├── 100.atr
│   └── ...
│
├── ensemble_klasifikatorius.py
├── README.md
├── ekg_model_metadata.json
├── ...
│
├── images/
│   ├── prt2_roc_curves.png
│   ├── prt2_metrics_comparison.png
│   └── ...
```

---

# Darbo paskirtis

Darbas buvo skirtas:

- medicininių signalų analizei,
- mašininio mokymosi tyrimams,
- EKG klasifikavimo demonstracijai.

---

# Autorė

Vasarė Pratuzaitė

Vilniaus universiteto Matematikos ir informatikos fakultetas

