# Apache Spark mit Jupyter-Notebook

## Tutorial:
### Starten der Spark Architektur in Docker:
`docker-compose up --build --scale spark-worker=3 -d` 

Falls man die Anzahl der Worker modifizieren möchte, dann muss man
den `spark-worker` Parameter ändern.

**Wichtig:** Der erste Build kann etwas länger Dauern (ca. 10 min),
da die Images noch gebuildet werden müssen.

### Wichtige Links:
Link für Jupyter-Notebook: \
`http://localhost:8888`

Link für den Spark-Master Node: \
`http://localhost:9090`

### Weitere Infos:
Der Ordner `notebooks` ist mit dem Jupyter-Notebook Container
verlinkt. Falls jemand ein Notebook hoch- oder herunterladen möchte,
dann kann dieser Ordner im Repository genutzt werden. Zusätzlich
befindet sich ein Template, wo sich ein Beispiel-Code befindet,
der Jupyter-Notebook Code in Spark-Cluster ausführen lässt.
