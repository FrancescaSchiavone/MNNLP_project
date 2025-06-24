# Licenza e Riferimenti

## Dati (EVALITA)

I dati utilizzati in questo progetto derivano da risorse pubbliche fornite nell'ambito di competizioni e studi accademici, in particolare:

- **EVALITA 2023 PoliticIT – Political Ideology Detection in Italian Texts** – Task di rilevamento dell’ideologia politica in testi italiani, organizzato nell’ambito di EVALITA 2023.  
  Maggiori informazioni e dataset sono disponibili su:  
  {https://codalab.lisn.upsaclay.fr/competitions/8507}

- Studi accademici correlati:  
  - García-Díaz, J. A., Colomo-Palacios, R., & Valencia-García, R. (2022). *Psychographic traits identification based on political ideology: An author analysis study on Spanish politicians’ tweets posted in 2020*. Future Generation Computer Systems, 130(1), 59-74.  
  - García-Díaz, J. A., Jiménez Zafra, S. M., Martín Valdivia, M. T., García-Sánchez, F., Ureña López, L. A., & Valencia García, R. (2022). *Overview of PoliticEs 2022: Spanish Author Profiling for Political Ideology*. Procesamiento del Lenguaje Natural, 69, 265-272.

Questi dataset e studi sono soggetti alle rispettive licenze e regolamenti delle competizioni e delle pubblicazioni originali.

## Codice (Hugging Face Transformers)

Il progetto include codice adattato dal repository ufficiale di [Hugging Face Transformers](https://github.com/huggingface/transformers): è stato utilizzato lo script -'run_glue_no_trainer.py'
{https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py}
Questi script sono rilasciati sotto **Apache License 2.0**. Per maggiori dettagli sulla licenza consultare il file LICENSE presente nel repository Hugging Face Transformers.

## Modelli pre-addestrati utilizzati

Per l’estrazione delle caratteristiche testuali e la classificazione, sono stati utilizzati modelli pre-addestrati disponibili su Hugging Face Hub, in particolare:

- roberta-base-italian (https://huggingface.co/osiria/roberta-base-italian)`

Questi modelli sono distribuiti con licenze specificate nelle loro pagine ufficiali su Hugging Face.

Si invita a consultare le pagine dei modelli per informazioni dettagliate sulle licenze e sulle condizioni d’uso.







