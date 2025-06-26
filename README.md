# **Fine-tuning BERT (Politic ideology detection)**
Il presente lavoro è un fine-tuning del modello BERT 'nome del modello' per un task di *text classification*: Politic Ideology Detection, binary classification \(*let/right*\). Il progetto cerca di riprodurre uno dei task presenti nell'edizione 2023 di Evalita, utilizzando i risultati ottenuti alla competizione come *benchmark*. Per maggiori dettagli sul task e sul dataset utilizzato, si veda il file *report.pdf* nella cartella \report.

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
  - fine_tuning.ipynb = collegamento con il foglio colab in cui è stato addestrato e testato il modello.
- \output:
  - \output training = cartella contenente il file *all_result.json* in cui è presente la metrica di accuratezza alla fine della fase di training sul *validation set*.
  - \output evaluation:
    - *prediction.csv* = predizioni realizzate dal modello sul test set esclusivamente rispetto alla colonna *'binary_ideology'*
    - *metriche.txt* = contiene le metriche di *accuracy*, *precision*, *recall*, *F1-score* e il *Classification Report*
    - *confusion_matrix.jpg* = matrice di confusione. 
