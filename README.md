# Sentimentanalyse
*Prosjekt 2 i HON2200*

Av Henrikke Gedde Rustad, Tor Magnus Næsset og Didrik Sten Ingebrigtsen

Vi ser på [NoReC](https://github.com/ltgoslo/norec)-datasettet, som har flere titalls tusen anmeldelser av blant annet filmer, serier og bøker fra norske aviser. Med seg har de alle et terningkast, og dette bruker vi en [fasttext](https://fasttext.cc)-modell til å predikere. Så drøfter vi hvordan kjønn spiller inn i dette. I `reports/report.ipynb` kan dere finne en gjennomgang av metode, og hvordan vi kommer fram til våre funn, og i `reports/report.pdf` har vi en rapport som forklarer hvordan vi tolker disse funnene, og hvorfor vi i det hele tatt har vært interessert i å stille de spørsmålene vi stiller.

Vi benytter oss av biblioteket Spacy, og [språkmodellen for bokmål som er tilgjengelig der](https://spacy.io/models/nb). For å laste ned den minste varianten, som er på 15MB og er den vi har brukt, kjør kommandoen:
`python -m spacy download nb_core_news_sm`. 

For å kjøre koden i `reports/report.ipynb` må dere først formattere dataen fra NoReC-datasettet på en måte fasttext takler, og dele datasettet opp i setninger for å kunne gjøre noen analyser. Dette kan dere gjøre ved å kjøre `sentiment.py` som er i rot-mappen. På en bærbar datamaskin tar dette noe i størrelsesorden en halvtime.
