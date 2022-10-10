# Control of a Genetic Toggle Switch

Author:
* Lorenzo Pagliara <l.pagliara5@studenti.unisa.it>

***
### Folder organisation
The project folder is organised as follows:

```
ToggleSwitchControl
└─── FPD
│   └─── data
│   └─── data
│   └─── data
│   └─── data
│   └─── data
│   └─── data
│   └─── data
│   └─── data
└─── Libraries
|
└─── MPC
|
└─── Simulations
│   └─── transcriptional_regulation.slx
│   └─── transcriptional_regulation_observer.slx
|   └─── transcriptional_regulation_kalman_filter.slx
|   └─── transcriptional_regulation_model.mlx
|   └─── transcriptional_regulation_fbl_controller.mlx   
└─── Report.pdf
└─── README.md
```

|File|Descrizione|
|:---|:---| 
|`transcriptional_regulation.mlx`| É il file principale per l'esecuzione di tutte le simulazioni.|
|`TranscriptionalRegulation.mat`| É il file contenente la configurazione del plugin Phase Plane per ottenere il diagramma di fase del sistema in esame.|
|`transcriptional_regulation.slx`| É il file Simulink contenente l'intero schema di retroazione del sistema.|
|`transcriptional_regulation_observer.slx`| É il file Simulink contenente l'intero schema di retroazione del sistema, con integrato l'osservatore.|
|`transcriptional_regulation_kalman_filter.slx`| É il file Simulink contenente l'intero schema di retroazione del sistema, con integrato il filtro di Kalman.|
|`transcriptional_regulation_model.mlx`| É il file contenente la funzione Matlab che implementa il modello matematico del sistema.|
|`transcriptional_regulation_fbl_controller.mlx`| É il file contenente la funzione Matlab che implementa il controllore non lineare del sistema.|

***
### Come eseguire lo script principale
Prima di eseguire lo script Matlab  `./transcriptional_regulation.mlx` occorre aggiungere al path di Matlab la cartella `Simulink`, con tasto destro sulla stessa:

```
Add to Path -> Selected Folders and Subfolders
```
L'esecuzione dello script avviene semplicemente selezionando il pulsante:

```
Run
```

***

### Diagramma di fase
Per visualizzare il diagramma di fase del sistema:

1. scaricare [PhasePlane](https://it.mathworks.com/matlabcentral/fileexchange/91705-phase-plane-and-slope-field-apps?s_tid=srchtitle_Phase%2520Plane_2);
2. avviare PhasePlane da Apps;
3. caricare il sistema da:

    ```
    Custom library -> load
    ```
    e selezionare il file `./Phase_Portrait/TranscriptionalRegulation.mat`;
4. impostare i parametri del sistema:

    ```
    Custom library -> RegolazioneTrascrizionale.
    ```

***

### Guida agli schemi Simulink
Lo schema di controllo realizzato in Simulink è stato ottenuto effettuando una divisione logica in blocchi dei vari elementi dello schema di retroazione. In tutti e tre gli schemi sono presenti:

* un blocco *Model* contenente il modello matematico del sistema, realizzato mediante la funzione Matlab presente nel file `transcriptional_regulation_model.mlx`. Nello schema di retroazione con observer e quello con il Kalman filter, all'interno del modello è presente un ulteriore blocco che realizza rispettivamente l'osservatore e il filtro;
* un blocco *FBL Controller - LQR* che implementa la legge di controllo non lineare, costituita da una parte lineare ottenuta mediante LQR e una parte non lineare implementata mediante la funzione Matlab presente nel file `transcriptional_regulation_fbl_controller.mlx`;
* un blocco *FBL Controller - PP* che implementa la legge di controllo non lineare, costituita da una parte lineare ottenuta mediante pole placement e una parte non lineare implementata mediante la funzione Matlab presente nel file `transcriptional_regulation_fbl_controller.mlx`.

I due blocchi controller sono collegati al sistema mediante uno switch guidato da una variabile `control`. 

Lo script principale è stato implementato in maniera tale da poter aprire gli schemi Simulink in maniera automatica ed effettuare la simulazione utilizzando prima l'uno e poi l'altro controllore, semplicemente modificando il valore della variabile `control`. I risultati di ciascuna simulazione sono poi inviati al workspace dello script che si occuperà, in maniera del tutto automatica, di effettuare il plot della risposta a ciclo chiuso del sistema, mettendo in comparazione quelle ottenute nelle stesse condizioni operative (no observer - no kalman filter, observer, kalman filter) con i due controllori.

***

### Note

1. Le versioni Matlab e Simulink utilizzate sono:

    * MATLAB  Version 9.11 (R2021b);
    * Simulink Version 10.4 (R2021b).
2. L'esecuzione dello script principale potrebbe richiedere un po' di tempo.
