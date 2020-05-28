# PoI Detection and Next Place Prediction

This is a python implementation of the two proposed approaches for [PoI Detection](http://sbrc2019.sbc.org.br/wp-content/uploads/2019/05/sbrc2019.pdf) and Next Place Prediction (coming soon). I developed these papers during my research at the Federal University of Viçosa, Brazil.

Research group website: [Nesped](http://www.nesped.caf.ufv.br/)

## Organization

The code has a job-based structure, which is helpful for future conversion to Pyspark.

| Folder/file | Comment |
| ------ | ------ |
| run.sh | It includes the configurations of the selected job |
| main.py | It calls the job set by run.sh |
| configuration | Set the algorithms parameters here |
| extractor | Codes that extract data from a specific source |
| foundation | It contains files of general purpose |
| domain | It contains the domain of each job. Each domain is responsible for calling the respective extractor and loader |
| job | It includes each job file |
| loader | It includes the loader type file which is applied to write the found results into a file using the specified structure |
| model | It contains files of classes that model objects |

## How to run?

First, install the requirements:

    pip install -r requirements.txt

Sencond, modify the run.sh file considering the directory where you have placed:

 - Your users' steps dataset (***users_steps_filename*** parameter)
 - Where you wanna save the detected PoIs (***poi_detection_filename*** parameter).
 - Where you wanna save the classified PoIs (***poi_classification_filename*** parameter).
 - The ground truth of the dataset (***ground_truth*** parameter)

There are three "execution options": 
 - "find_pois": applied to generate the users' pois.
 - "validate": applied to validate the generated users' pois.
 - "find_pois_and_validate": applied to run together the two previous options.

Finally, run the bash script informing a execution option argument:

    bash run.sh "find_pois_and_validate"

    
---

## License

The license is free. If this work was helpful for you, i would appreciate a citation in scientific publications. 

Bibtex entry of the PoI Detection paper:

    @ARTICLE {capanema2019identificacao,
    author  = {Cláudio G. S. Capanema and Fabrício A. Silva and Thais R. M. B. Silva},
    title   = {Identificação e Classificação de Pontos de Interesse Individuais com Base em Dados Esparsos},
    journal = {Anais do XXXVII Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos - SBRC 2019},
    year    = {2019},
    pages   = {16-29},
    month   = {Maio}
    }



