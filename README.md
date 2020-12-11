# Modo de usar
###### Parâmetros:
* ```--imagesDir -id``` : Diretório raiz do dataset.
* ```--model -m```: Modelo a ser utilizado, tendo como opções ``darknet19`` ou ``darkCovidNet``.
* ```--epochs -e```: Quantidade de épocas a serem executadas.
* ```--pth_path -pth```: Diretório para salvar os modelos treinados.
* ```--log_path -log```: Diretório para salvar os logs.

###### Exemplo:
```
  python train.py -id 'dataset' -m darknet19 -e 5 -pth ./trained_models -log ./logs
```
# Artefatos
* Relatório disponível em: [Relatório.pdf](https://github.com/CamAndrade/visao-computacional/blob/master/Relat%C3%B3rio.pdf)
* Vídeo disponível em: [Apresentação](https://www.youtube.com/watch?v=catqMgMjcFY&feature=youtu.be)

# Referências
* Dataset extraído de [COVID-19](https://github.com/muhammedtalo/COVID-19/tree/master/X-Ray%20Image%20DataSet).

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
