<h1 align="left">
EKG aritmijų klasifikavimas naudojant ansamblinius metodus
</h1>

<p align="left">
Mašininio mokymosi projektas EKG signalų analizei ir aritmijų aptikimui naudojant MIT-BIH Arrhythmia Database duomenų rinkinį.
</p>

---

# Projekto aprašymas

Šiame projekte sukurta sistema, kuri automatiškai klasifikuoja EKG širdies dūžius į dvi klases:

- normalius dūžius,
- aritminius dūžius.

Modeliui kurti naudoti šie metodai:

- Logistic Regression,
- Random Forest,
- Soft Voting Ensemble.

Klasių disbalanso problemai mažinti pritaikytas SMOTE metodas.

---

# Projekto tikslas

Taikant signalų apdorojimo ir mašininio mokymosi metodus sukurti modelį, galintį klasifikuoti elektrokardiogramos signalus ir nustatyti, ar širdies ritmas atitinka nustatytas normas.

---

# Darbo uždaviniai:
1. Išanalizuoti pagrindines medicininių signalų rūšis ir jų savybes.
2. Apžvelgti biologinių signalų apdorojimo metodus.
3. Aptarti mašininio mokymosi ir giliojo mokymosi metodus, taikomus laiko eilučių analizei.
4. Pristatyti sukurtą EKG signalų klasifikavimo modelį.
5. Įvertinti modelio veikimo tikslumą ir praktinį pritaikomumą diagnostikoje.
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

# Naudotos bibliotekos

- NumPy
- Pandas
- Scikit-learn
- imbalanced-learn
- WFDB
- Matplotlib
- Joblib

---

# Projekto paskirtis

Projektas skirtas:

- medicininių signalų analizei,
- mašininio mokymosi tyrimams,
- EKG klasifikavimo demonstracijai.

---

# Autorė

Vasarė Pratuzaitė
Vilniaus universiteto Matematikos ir informatikos fakultetas

