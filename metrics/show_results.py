import json
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def treino_validacao_plot(arquivos_csv, title):
    metricas_cores = [
        ('train_loss', 'blue'),
        ('val_acc', 'red'),
        ('val_loss', 'orange')
    ]
    for metrica, cor in metricas_cores:
        count = 1
        for fold in arquivos_csv:
            plt.rcParams.update({'font.size': 100})
            plt.figure(figsize=(50, 25))

            darknet19_dir = os.path.join(fold[0], metrica + '.csv')
            darkCovidNet_dir = os.path.join(fold[1], metrica + '.csv')

            darknet19_df = pd.read_csv(darknet19_dir)
            darkCovidNet_df = pd.read_csv(darkCovidNet_dir)

            darknet19_df['Value'].plot(label='darknet19', color=cor, legend='darknet19', linewidth=10)
            darkCovidNet_df['Value'].plot(label='darkCovidNet', color=cor, linestyle='dashed', legend='darkCovidNet', linewidth=10)

            if title:
                plt.title('fold_' + str(count) + ': ' + metrica)
            print('fold_' + str(count) + ': ' + metrica)
            count += 1
            plt.grid()

            plt.xlabel('Épocas')
            plt.ylabel('Valor')

            plt.subplots_adjust(top=0.997, bottom=0.159, right=0.999, left=0.1)

            plt.show()
            plt.close()


def matriz_confusao_plot(title, labels, matriz_confusao):
    """https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7"""
    plt.rcParams.update({'font.size': 50})

    cm = matriz_confusao
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = '%.1f%%\n(%d)' % (p, c)

    cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(16, 17))
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='Greys', cbar=False, linewidths=5, linecolor='black')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.subplots_adjust(top=0.93, bottom=0.262, right=0.993, left=0.27)

    plt.show()
    plt.close()


def teste_plot(arquivos_json):
    for key in arquivos_json:
        fold_count = 1
        print(key)
        print('Fold\t', 'Sens.\t', 'Espe.\t',
                      'Prec.\t', 'Me-F.\t', 'Acur.')

        matriz_sobreposta = list()

        for fold in arquivos_json[key]:
            with open(fold, 'r') as json_fold:
                metricas = json.load(json_fold)

                rotulos = metricas['classes'].keys()
                matriz_confusao = np.array(metricas['matriz_confusao'])
                sensibilidade = metricas['sensibilidade']
                especificidade = metricas['especificidade']
                precisao = metricas['precisao']
                medida_f = metricas['medida_f']
                acuracia = metricas['acuracia']

                print(fold_count, '\t',
                      round(sensibilidade * 100, 2), '\t',
                      round(especificidade * 100, 2),'\t',
                      round(precisao * 100, 2), '\t',
                      round(medida_f * 100, 2), '\t',
                      round(acuracia * 100, 2))

                matriz_sobreposta.append(matriz_confusao)
                matriz_confusao_plot('Fold-' + str(fold_count), rotulos, matriz_confusao)

                fold_count += 1
        print()

        matriz_sobreposta = np.array(matriz_sobreposta).sum(axis=0)
        matriz_confusao_plot('Sobreposta', rotulos, matriz_sobreposta)


def main():
    # melhores resultados obtidos, levando em consideracao o desempenho da darkCovidNet
    # n_classes = 2
    # seed = 1
    # n_classes = 3
    # seed = 5
    n_classes = 3
    seed = 5

    # arquivos para comparar o desempenho no treino e validacao da darknet19 com a darkCovidNet
    arquivos_csv = [
        ('logs' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_0/version_0',
         'logs' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_0/version_0'),
        ('logs' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_1/version_0',
         'logs' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_1/version_0'),
        ('logs' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_2/version_0',
         'logs' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_2/version_0'),
        ('logs' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_3/version_0',
         'logs' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_3/version_0'),
        ('logs' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_4/version_0',
         'logs' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_4/version_0')
    ]

    # arquivos para comparar o desempenho no teste da darknet19 com a darkCovidNet
    arquivos_json = dict(
        darkCovidNet=
        ('results' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_0/metricas.json',
         'results' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_1/metricas.json',
         'results' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_2/metricas.json',
         'results' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_3/metricas.json',
         'results' + str(n_classes) + '/' + str(seed) + '/darkCovidNet/fold_4/metricas.json'),
        darknet19=
        ('results' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_0/metricas.json',
         'results' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_1/metricas.json',
         'results' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_2/metricas.json',
         'results' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_3/metricas.json',
         'results' + str(n_classes) + '/' + str(seed) + '/darknet19/fold_4/metricas.json')
    )

    treino_validacao_plot(arquivos_csv, False)
    teste_plot(arquivos_json)


if __name__ == '__main__':
    main()
