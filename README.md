<h1 align="center">
EKG aritmijų klasifikavimas naudojant ansamblinius metodus
</h1>

<p align="center">
Mašininio mokymosi projektas, skirtas EKG signalų analizei ir aritmijų aptikimui naudojant MIT-BIH Arrhythmia Database duomenų rinkinį.
</p>

---

# Projekto aprašymas

Šiame projekte sukurta EKG signalų klasifikavimo sistema, kuri automatiškai atskiria normalius ir aritminius širdies dūžius.

Modelio kūrimui naudoti keli klasifikavimo metodai bei jų sujungimas į ansamblinį modelį:

- Logistic Regression
- Random Forest
- Soft Voting Ensemble

Papildomai pritaikytas SMOTE metodas klasių disbalanso mažinimui.

---

# Projekto tikslas

Sukurti klasifikavimo sistemą, kuri:

- automatiškai apdorotų EKG signalus,
- išskirtų dūžius iš MIT-BIH įrašų,
- klasifikuotų normalius ir aritminius signalus,
- sumažintų klasių disbalanso problemą,
- pateiktų rezultatų analizę grafikų forma.

---

# Naudoti metodai

## Logistic Regression

Logistinė regresija naudota kaip bazinis klasifikatorius.

Pagrindinės savybės:

- greitas mokymas,
- interpretuojami rezultatai,
- stabilus darbas su standartizuotais požymiais.

---

## Random Forest

Random Forest metodas sudarytas iš daugelio sprendimų medžių.

Pagrindinės savybės:

- geba modeliuoti netiesinius ryšius,
- atsparus triukšmui,
- gerai veikia su sudėtingais signalų požymiais.

---

## Soft Voting Ensemble

Galutinis modelis sukurtas naudojant ansamblinį metodą.

Ansamblio sudėtis:

```python
VotingClassifier(
    estimators=[
        ("logreg", logistic_model),
        ("rf", rf_model)
    ],
    voting="soft",
    weights=[1.0, 3.0]
)
```

Soft Voting metodas sujungia kelių modelių prognozes pagal jų tikimybes.

Tai leidžia:
- sumažinti pavienių modelių klaidas,
- pagerinti stabilumą,
- padidinti klasifikavimo tikslumą.

---

# Klasių disbalansas

MIT-BIH duomenų rinkinyje normalių dūžių yra žymiai daugiau nei aritminių.

Pradinė distribucija:

| Klasė | Kiekis |
|---|---|
| Normalūs dūžiai | 97 635 |
| Aritminiai dūžiai | 11 833 |

Toks disbalansas mažina modelio gebėjimą aptikti retesnę klasę.

---

# SMOTE metodas

Klasių balansavimui naudotas SMOTE metodas.

SMOTE generuoja sintetinius mažumos klasės pavyzdžius.

Po balansavimo:

```text
Normalūs: 73218
Aritminiai: 73218
```

SMOTE pagerino:
- Recall,
- F1-score,
- Balanced Accuracy.

---

# Požymių ištraukimas

Iš kiekvieno EKG dūžio ištraukiami statistiniai ir signaliniai požymiai:

- vidurkis,
- standartinis nuokrypis,
- energija,
- RMS,
- peak-to-peak amplitudė,
- skewness,
- kurtosis,
- pirmos ir antros išvestinės,
- zero crossings,
- QRS pločio aproksimacija.

---

# Naudotas duomenų rinkinys

## MIT-BIH Arrhythmia Database

Naudoti failų tipai:

```text
.dat  -> EKG signalas
.hea  -> metaduomenys
.atr  -> anotacijos
```

Projektui naudoti 48 pilni įrašai.

---

# Rezultatai

## Logistic Regression

| Metrika | Reikšmė |
|---|---|
| Balanced Accuracy | 0.6415 |
| F1-score | 0.2674 |
| Precision | 0.1962 |
| Recall | 0.4196 |
| ROC AUC | 0.6024 |

---

## Random Forest

| Metrika | Reikšmė |
|---|---|
| Balanced Accuracy | 0.6920 |
| F1-score | 0.3499 |
| Precision | 0.2730 |
| Recall | 0.4871 |
| ROC AUC | 0.7967 |

---

## Ensemble modelis

| Metrika | Reikšmė |
|---|---|
| Balanced Accuracy | 0.6806 |
| F1-score | 0.3442 |
| Precision | 0.2764 |
| Recall | 0.4562 |
| ROC AUC | 0.7647 |

---

# Sugeneruojami grafikai

Programa automatiškai sugeneruoja:

- klasių pasiskirstymo grafiką,
- SMOTE balansavimo grafiką,
- confusion matrix grafikus,
- ROC kreives,
- modelių metrikų palyginimo grafikus.

---

# Rezultatų pavyzdžiai

## ROC kreivės

![ROC](images/prt2_roc_curves.png)

---

## Ensemble confusion matrix

![CM](images/prt2_confusion_matrix_ensemble.png)

---

## Modelių metrikų palyginimas

![Metrics](images/prt2_metrics_comparison.png)

---

# Projekto struktūra

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
├── ekg_ensemble_model.joblib
├── ekg_model_metadata.json
│
├── images/
│   ├── prt2_roc_curves.png
│   ├── prt2_metrics_comparison.png
│   └── ...
```

---

# Paleidimas lokaliai

## Reikalingos bibliotekos

```bash
pip install wfdb imbalanced-learn joblib matplotlib pandas numpy scikit-learn
```

---

## Programos paleidimas

```bash
python ensemble_klasifikatorius.py
```

---

# Paleidimas Google Colab aplinkoje

## 1. Bibliotekų įdiegimas

```python
!pip install wfdb imbalanced-learn joblib matplotlib pandas numpy scikit-learn
```

---

## 2. Dataset įkėlimas

```python
from google.colab import files
uploaded = files.upload()
```

---

## 3. Dataset išarchyvavimas

```python
!unzip dataset.zip -d /content/dataset
```

---

## 4. Notebook blokų paleidimas

Po paleidimo automatiškai sugeneruojami:
- modeliai,
- JSON metadata failas,
- grafikai.

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
- EKG klasifikavimo demonstracijai,
- edukaciniams tikslams.

Sistema nėra skirta klinikinei diagnostikai.

---

# Autorė

Vasarė Pratuzaitė

Vilniaus universitetas
Duomenų bazės nuoroda: https://www.kaggle.com/code/gregoiredc/arrhythmia-on-ecg-classification-using-cnn 
