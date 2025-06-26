# **Fine-tuning PoliticIT**
![Python](https://img.shields.io/badge/Lang-Python-blue)
Il presente lavoro è un fine-tuning del modello RoBERTa (xlm-roberta-base) per un task di *text classification*: **Politic Ideology Detection**, binary classification \(*let/right*\) con l'ausilio dello script *run_glue_no_trainer.py* messo a disposizione da Hugging Face.\
Il progetto cerca di riprodurre uno dei task presenti nell'edizione 2023 di Evalita, utilizzando i risultati ottenuti alla competizione come *benchmark*. Per maggiori dettagli sul task e sul dataset utilizzato, si veda il file *Gorzoni_Schiavone_MNNLP.pdf* nella cartella \report.

## Documentation
La repository è così articolata:
- \Data = cartella contenente i dati utilizzati per il fine-tuning:
  - training set file: *politicIT_phase_2_train.csv*
  - validation set file: *politicIt_phase_2_validation.csv*
  - test set file: *politicIt_phase_2_test_codalab.csv*
  - Split dataset.ipynb = collegamento con il foglio colab in cui è stato diviso il file train in *training set* e *validation set*
- \src = cartella contenente i file relativi al fine-tuning:
  - run_glue_no_trainer-4.py = script adattato dall'originale al task qui presente
  - evaluate_model.py = file utilizzato per la valutazione del modello sui dati del *test set*
  - roberta.ipynb = collegamento con il foglio colab in cui è stato addestrato e testato il modello.
- \output:
  - \output training = cartella contenente il file *all_result.json* in cui è presente la metrica di accuratezza alla fine della fase di training sul *validation set*.
  - \output evaluation:
    - *metriche.txt* = contiene le metriche di *accuracy*, *precision*, *recall*, *F1-score* e il *Classification Report*
    - *confusion_matrix.jpg* = matrice di confusione.
- \report:
  - Gorzoni_Schiavone_MNNLP.pdf = Il report a corredo del progetto analizza la struttura del Dataset scelto, passa in rassegna i dettagli tecnici del fine-tuning: dal preprocessing del dataset, alle modifiche effettuate sullo script *run\_glue\_no\_trainer.py*, alla scelta degli iperparametri.
- file README.md 
- file DATA_LICENCE.md
