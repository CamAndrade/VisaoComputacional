# Modo de usar
###### Parâmetros:
* ```--imagesDir -id``` : Diretório raiz do dataset.
* ```--model -m```: Modelo a ser utilizado, tendo como opções ``darknet19`` ou ``darkCovidNet``.
* ```--epochs -e```: Quantidade de épocas a serem executadas.

###### Exemplo:
```
  python train.py -id 'dataset' -m darknet19 -e 5
```

# Referências
Dataset extraído de [COVID-19](https://github.com/muhammedtalo/COVID-19/tree/master/X-Ray%20Image%20DataSet).

```
@inproceedings{redmon2017yolo9000,
  title={YOLO9000: better, faster, stronger},
  author={Redmon, Joseph and Farhadi, Ali},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7263--7271},
  year={2017}
}

@article{ozturk2020automated,
  title={Automated detection of COVID-19 cases using deep neural networks with X-ray images},
  author={Ozturk, Tulin and Talo, Muhammed and Yildirim, Eylul Azra and Baloglu, Ulas Baran and Yildirim, Ozal and Acharya, U Rajendra},
  journal={Computers in Biology and Medicine},
  pages={103792},
  year={2020},
  publisher={Elsevier}
}

```