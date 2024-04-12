import shapefile, csv
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import sys
from seg2 import Segreg
import math
import scipy.stats as st
import json

class Utils():

    def __init__(self):
        self.data = {}
        self.dissi = {}
        self.entr = {}
        self.iH = {}
        self.expo_iso = {}
        self.list_csv = []
        self.list_color = []

    def rotate90Clockwise(self, A, transpose = False):
        N = len(A[0])
        for i in range(N // 2):
            for j in range(i, N - i - 1):
                temp = A[i][j]
                A[i][j] = A[N - 1 - j][i]
                A[N - 1 - j][i] = A[N - 1 - i][N - 1 - j]
                A[N - 1 - i][N - 1 - j] = A[j][N - 1 - i]
                A[j][N - 1 - i] = temp
        if transpose:
            return np.transpose(A)
        else:
            return A

    def createCSV(self, path_to_read, path_to_write, inputfiles):
        for i in inputfiles:
            pathread = path_to_read + i + '.txt'
            pathwrite = path_to_write + i + '.csv'
            read_file = pd.read_csv(pathread, delimiter=' ', header=None)
            read_file = read_file.drop(read_file.columns[-1],axis=1)
            read_file.to_csv(pathwrite, index=None, header=None)

    def createSHP(self, path_to_read, path_to_write, inputfiles):
        for inst in inputfiles:

        # path_to_read = '/content/gdrive/My Drive/M.Sc./data/comparacao5x5/' + inst
            path = path_to_write + inst

            dots_shp = shapefile.Writer(path)
            dots_shp.autoBalance = 1

            dots_shp.field("AGENT_ID", "C")
            dots_shp.field("AGENT_RED", "C")
            dots_shp.field("AGENT_YELLOW", "C")
            dots_shp.field("AGENT_BLUE", "C")
            dots_shp.field("AGENT_CYAN", "C")
            # dots_shp.field("AGENT_GRAY", "C")

            name = path_to_read + inst + '.csv'
            espacamento = 50
            nextX = espacamento
            nextY = -espacamento
            with open(name, 'r') as csvfile:
                id = 0
                reader = csv.reader(csvfile, delimiter=',')
                reader = list(self.rotate90Clockwise(list(reader), True))
                reader = self.rotate90Clockwise(reader)
                # skip the header
                # next(reader, None)
                i = 0
                for row in reader:
                    j = 0
                    for elem in row:

                        # if elem == '0' or elem =='1' or elem == '2' or elem == '3' or elem == '4':
                        agent_id = id
                        agent_red = 0
                        agent_yellow = 0
                        agent_blue = 0
                        agent_cyan = 0
                        # agent_gray = 0
                        if elem == '1':
                            agent_red = 1
                        elif elem == '2':
                            agent_yellow = 1
                        elif elem == '3':
                            agent_blue = 1
                        elif elem == '4':
                            agent_cyan = 1

                        x = espacamento*i
                        y = espacamento*j
                        dots_shp.polym([ [[y, -x, float(elem)], [y, -x+nextX, float(elem)], [y+nextY, -x+nextX, float(elem)], [y+nextY, -x, float(elem)]]])

                        # dots_shp.pointm(float(j),-(float(i)), float(elem))
                        # add attribute data
                        dots_shp.record(agent_id, agent_red, agent_yellow, agent_blue, agent_cyan)
                        j += 1
                        id += 1

                    i += 1

            dots_shp.close()

    def dataCleaner(self, path_to_read, listNames):
        dict = {}
        ind = 0
        # for inst in listNames:
        info = []
        name = path_to_read + listNames + '_global.csv'
        with open(name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = -1
            lista_matrix = []
            # print('here')
            for row in reader:
                i += 1
                # print(i, row)
                n = re.findall(r"[-+]?\d*\.\d+|\d+", str(row))
                if i < 3:
                    info.append(n)
                if i == 3:
                    continue
                if i > 3:
                    lista_matrix.append(n)

            info.append(lista_matrix)
            dict[ind] = info
            ind += 1

        return dict

    def list_flatter(self, lista):

        l2 = []
        for item in lista:
            i2 = [x for x in item if x]
            l2.append(i2)

        flat_list = [item for sublist in l2 for item in sublist]
        # print(flat_list)
        flat_list = [float(i) for i in flat_list]

        return flat_list

    def separar(self, lista):
        l = []
        l1 = []
        l2 = []
        l3 = []
        for x in range(0, len(lista)):
            l.append(lista[x][0])
            l1.append(lista[x][1])
            l2.append(lista[x][2])
            l3.append(lista[x][3])

        l = [float(i) for i in l]
        l1 = [float(i) for i in l1]
        l2 = [float(i) for i in l2]
        l3 = [float(i) for i in l3]

        return [l, l1, l2, l3]

    def dumpDict(self, dict2, path, name, dataset):
        outpath = path + dataset + '_' + name
        outjson = outpath + '.json'
        json2 = json.dumps(dict2)
        f = open(outjson,"w")
        f.write(json2)
        f.close()

        outcsv = outpath + '.csv'
        w = csv.writer(open(outcsv, "w"))
        for key, val in dict2.items():
            w.writerow([key, val])

    def plot(self, lista_plot, list_methods, label_t, color, alpha, xtick, tick, path):
        # multiple line plot
        fig = plt.figure()

        plt.plot(lista_plot[0], marker='',  color=color[0], linewidth=2, linestyle = '-', label='blue', alpha=alpha)
        plt.plot(lista_plot[1], marker='', color=color[1], linewidth=2, label='red', linestyle= '--' , alpha=alpha)
        plt.plot(lista_plot[2], marker='', color=color[2], linewidth=2,  label="yellow", linestyle = '-.', alpha=alpha)
        plt.plot(lista_plot[3], marker='', color=color[3], linewidth=2, label="cyan", alpha=alpha)
        plt.xticks(range(0, len(xtick), 1), labels=xtick)
        plt.yticks(tick)
        plt.xlabel('Runs')
        plt.ylabel('Isolamento/Exposição')
        plt.title(label_t, wrap=True)
        plt.grid()
        plt.legend(list_methods, loc=1)
        # plt.show()
        a = label_t.replace(' ', '_')
        a = a.replace('/', '_')
        # plt.tight_layout()

        # print(a)
        name = path + a + '.png'

        plt.savefig(name)
        plt.close(fig)

    def plotAcc(self, data_acc, steps, label, ytick, ylab, path, xtick):

        # data_acc = np.multiply(data_acc, 10)
        # print(data_acc)
        c = range(len(data_acc))
        # print(c)
        fig = plt.figure(figsize=((12,7)))
        fig.add_subplot(122)
        ax = plt.axes()
        ax.plot(c, data_acc, 'k')
        plt.yticks(ytick)
        plt.xticks(range(0, len(xtick), 1), labels=xtick)
        # plt.xticks(rotation=90)
        plt.title(label)
        plt.ylabel(ylab)
        plt.xlabel("Runs")
        plt.grid()
        plt.tight_layout()

        # plt.show()
        name_l = label.replace(' ', '_')
        name = path + name_l + '.png'
        plt.savefig(name)
        plt.close(fig)

    def plotIntervalConfidence(self, title, listValue, ciValue, listMethods, path):

        n = len(listValue)
        fig = plt.figure()

        bars = []
        barsLabels = []
        barWidth = 0.3
        ci = []
        for i in range(0, n):
            bars.append(listValue[i])
            barsLabels.append(listMethods[i])
            ci.append(ciValue[i][1] - ciValue[i][0])

        # The x position of bars
        r1 = np.arange(len(bars))
        # r2 = [x + barWidth for x in r1]

        # Create blue bars
        plt.bar(r1, bars, width = barWidth, edgecolor = 'black', yerr=ci, ecolor= 'xkcd:mustard', capsize=7)

        # Create cyan bars
        # plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', yerr=yer2, capsize=7, label='sorgho')

        # general layout
        plt.xticks([r for r in range(len(bars))], listMethods)
        plt.ylabel('Intervalo de Confiança')
        plt.title(title, wrap=True)
        plt.tight_layout()

        # plt.legend()
        name_l = title.replace(' ', '_')
        name_l = name_l.replace('/', '_')
        name = path + name_l + '.png'
        plt.savefig(name)
        plt.close(fig)

        # Show graphic
        # plt.show()

    def processPlot2(self,listFiles, path, outpath, dirfile):

        for inst in listFiles:
            name = path + 'tabelas/' + inst + '.csv'

            listValue = []
            listMethods = []
            ciValue = []
            title = ''

            if 'segreg' in inst:
                title = 'Segregação no dataset ' + inst
            elif 'yellow' in inst:
                title = 'Exposição/Isolamento do agent yellow no dataset ' + inst
            elif 'blue' in inst:
                title = 'Exposição/Isolamento do agent blue no dataset ' + inst
            elif 'cyan' in inst:
                title = 'Exposição/Isolamento do agent cyan no dataset ' + inst
            elif 'red' in inst:
                title = 'Exposição/Isolamento do agent red no dataset ' + inst

            with open(name, 'r') as csvfile:

                reader = csv.reader(csvfile, delimiter=',')
                # print(reader)
                next(reader, None)
                for row in reader:
                    listValue.append(float(row[4]))
                    listMethods.append(row[0])
                    ciValue.append( [float(row[7]), float(row[8])] )
                    # print(row[4])

                self.plotIntervalConfidence(title, listValue, ciValue, listMethods, outpath)

    def pre_plots(self, lista, dict, j):

        aux = 10
        path = '/content/gdrive/My Drive/M.Sc./data/comparacao5x5/plots/'
        lista_diss = []
        lista_indexH = []
        title_diss = ''
        title_h = ''
        e = int(j/5)-1
        title_diss = 'Dissimilaridade do dataset ' + listN[e] + ' em relação ao bandwidth de 5 células'
        title_h = 'Indice H dos dataset ' + listN[e] + ' em relação ao bandwidth de 5 células'
        xtick = lista#[10,20,30,40,50,60,70,80,90,100]
        for i in range(j-5, j):

            lista_diss.append(dict[i][0])
            lista_indexH.append(dict[i][2])

        lista_diss = list_flatter(lista_diss)
        diss_max = max(lista_diss)
        diss_min = min(lista_diss)
        tick = np.linspace(diss_min-diss_min*0.1,diss_max+diss_max*0.1, num=10).tolist()
        plotAcc(lista_diss, 0.1, title_diss, tick, "Dissimilaridade", path, xtick)

        lista_indexH = list_flatter(lista_indexH)
        indexH_max = max(lista_indexH)
        indexH_min = min(lista_indexH)
        tick = np.linspace(indexH_min-indexH_min*0.1,indexH_max+indexH_max*0.1, num=10).tolist()
        plotAcc(lista_indexH, 0.1, title_h, tick, 'Indice H', path, xtick)

    def pre_plots_full(self, outpathplots, listDict, inputfiles, bandwidth, outresultpath, dirfile):

        dataset = dirfile#str(inputfiles[0][0:-3])
        # print(dataset)
        # print(inputfiles[0])
        # sys.exit(0)
        xtick = []
        lista_diss = []
        lista_entropy = []
        lista_indexH = []
        lista_blue = []
        lista_red = []
        lista_cyan = []
        lista_yellow = []
        title_diss = ''
        title_en = ''
        title_h = ''
        title_ex = ''
        title_exp = ''
        title_F1 = ''
        listDi = []
        listEntr = []
        listiH = []
        self.list_csv = []
        self.list_color = []
        latex_path = outresultpath + 'tabelas/'
        dict_path = outresultpath + 'dictionaries/'
        for i in range(0, len(listDict)):
            xtick.append(i)
            lista_diss.append(listDict[i][0][0])
            lista_entropy.append(listDict[i][0][1])
            lista_indexH.append(listDict[i][0][2])
            lista_red.append(listDict[i][0][3][0])
            lista_yellow.append(listDict[i][0][3][1])
            lista_blue.append(listDict[i][0][3][2])
            lista_cyan.append(listDict[i][0][3][3])

        title_diss = 'Dissimilaridade do dataset ' + dataset + ' em relação ao bandwidth de '+ str(int(bandwidth))+ ' células'
        title_en = 'Entropia do dataset ' + dataset + ' em relação ao bandwidth de '+ str(int(bandwidth))+ ' células'
        title_h =  'Indice H do dataset ' + dataset + ' em relação ao bandwidth de '+ str(int(bandwidth))+ ' células'
        title_ex = 'no dataset ' + dataset + ' em relação ao bandwidth '+ str(int(bandwidth))+ ' células'

        lista_diss = self.list_flatter(lista_diss)
        diss_max = max(lista_diss)
        diss_min = min(lista_diss)
        diss_median = np.median(lista_diss)
        diss_mean = np.mean(lista_diss)
        diss_var = np.var(lista_diss)
        diss_std = np.std(lista_diss)
        diss_ci = st.t.interval(alpha=0.95, df=len(lista_diss)-1, loc=np.mean(lista_diss), scale=st.sem(lista_diss))
        self.dissi = {'max': diss_max, 'min': diss_min, 'median': diss_median, 'mean': diss_mean, 'var': diss_var, 'std': diss_std, 'ci': diss_ci}
        self.data['dissimilaridade'] = self.dissi

        listDi = ['dissimilaridade', diss_max, diss_min, diss_median, diss_mean, diss_var, diss_std, diss_ci[0], diss_ci[1]]
        # sys.exit(0)
        # Prepare to plot the dissimilarity
        tick = np.linspace(diss_min-diss_min*0.1,diss_max+diss_max*0.1, num=10).tolist()
        self.plotAcc(lista_diss, 0.1, title_diss, tick, "Dissimilaridade", outpathplots, xtick)

        lista_entropy = self.list_flatter(lista_entropy)
        entropy_max = max(lista_entropy)
        entropy_min = min(lista_entropy)
        entropy_max = max(lista_entropy)
        entropy_min = min(lista_entropy)
        entropy_median = np.median(lista_entropy)
        entropy_mean = np.mean(lista_entropy)
        entropy_var = np.var(lista_entropy)
        entropy_std = np.std(lista_entropy)
        entropy_ci = st.t.interval(alpha=0.95, df=len(lista_entropy)-1, loc=np.mean(lista_entropy), scale=st.sem(lista_entropy))
        self.entr = {'max': entropy_max, 'min': entropy_min, 'median': entropy_median, 'mean': entropy_mean, 'var': entropy_var, 'std': entropy_std, 'ci': entropy_ci}
        self.data['entropia'] = self.entr
        listEntr = ['entropia', entropy_max, entropy_min, entropy_median, entropy_mean, entropy_var, entropy_std, entropy_ci[0], entropy_ci[1]]

        # prepare to plot the entropy
        tick = np.linspace(entropy_min-entropy_min*0.1,entropy_max+entropy_max*0.1, num=10).tolist()
        self.plotAcc(lista_entropy, 0.1, title_en, tick, "Entropia", outpathplots, xtick)

        lista_indexH = self.list_flatter(lista_indexH)
        indexH_max = max(lista_indexH)
        indexH_min = min(lista_indexH)
        indexH_max = max(lista_indexH)
        indexH_min = min(lista_indexH)
        indexH_median = np.median(lista_indexH)
        indexH_mean = np.mean(lista_indexH)
        indexH_var = np.var(lista_indexH)
        indexH_std = np.std(lista_indexH)
        indexH_ci = st.t.interval(alpha=0.95, df=len(lista_indexH)-1, loc=np.mean(lista_indexH), scale=st.sem(lista_indexH))
        self.iH = {'max': indexH_max, 'min': indexH_min, 'median': indexH_median, 'mean': indexH_mean, 'var': indexH_var, 'std': indexH_std, 'ci': indexH_ci}
        self.data['indexH'] = self.iH
        listiH = ['indexH', indexH_max, indexH_min, indexH_median, indexH_mean, indexH_var, indexH_std, indexH_ci[0], indexH_ci[1]]

        # prepare to plot index H
        tick = np.linspace(indexH_min-indexH_min*0.1,indexH_max+indexH_max*0.1, num=10).tolist()
        self.plotAcc(lista_indexH, 0.1, title_h, tick, 'Indice H', outpathplots, xtick)

        # preparar plot para cada run do mesmo cenario, 4 plots, 1 para cada tipo de agente
        list_methods = ['red', 'yellow', 'blue', 'cyan']
        color = ['red', 'xkcd:gold', 'blue', 'cyan']
        analise = [lista_red, lista_yellow, lista_blue, lista_cyan]

        for an in analise:
            lista_plot = self.separar(an)
            l = self.list_flatter(lista_plot)
            diss_max = max(l)
            diss_min = min(l)
            # print(diss_max, diss_min)
            tick = np.linspace(diss_min-diss_min*0.1,diss_max+diss_max*0.1, num=10).tolist()
            # print(tick)
            tick = [float(i) for i in tick]
            lista_plot0 = [lista_plot[0], lista_plot[1], lista_plot[2], lista_plot[3]]

            if an == lista_blue:
                title_exp = "Exposição/Isolamento do agente azul " + title_ex
                title_F1 = 'F1 em relação ao Agente Azul'
            if an == lista_red:
                title_exp = "Exposição/Isolamento do agente vermelho " + title_ex
                title_F1 = 'F1 em relação ao Agente Vermelho'
            if an == lista_yellow:
                title_exp = "Exposição/Isolamento do agente amarelo " + title_ex
                title_F1 = 'F1 em relação ao Agente Amarelo'
            if an == lista_cyan:
                title_exp = "Exposição/Isolamento do agente ciano " + title_ex
                title_F1 = 'F1 em relação ao Agente Ciano'


            # print(title_exp)
            self.plot(lista_plot0, list_methods, title_exp, color, 0.7, xtick, tick, outpathplots)


        # print(lista_red)
        listMeasures = ['Measure', 'max', 'min', 'median', 'mean', 'var', 'standart dev', 'ci_floor', 'ci_ceil']
        self.list_csv = [listMeasures, listDi, listEntr, listiH]
        c = 0
        dict2 = {}
        for group in analise:
            laux = []
            listdoGrupo = []
            dicti = {}
            nametab = 'Tab_' + list_methods[c]
            self.list_color.append([nametab, 'max', 'min', 'median', 'mean', 'var', 'standart dev', 'ci_floor', 'ci_ceil'])
            # print(self.list_color[0][0][4:])
            # sys.exit(0)
            for i in range(0, 4):
                listColor = []
                laux = [row[i] for row in group]
                laux = [float(item) for item in laux]

                # print(type(laux), type(laux[1]))
                # sys.exit(0)
                indexH_max = max(laux)
                indexH_min = min(laux)
                indexH_max = max(laux)
                indexH_min = min(laux)
                indexH_median = np.median(laux)
                indexH_mean = np.mean(laux)
                indexH_var = np.var(laux)
                indexH_std = np.std(laux)
                indexH_ci = st.t.interval(alpha=0.95, df=len(laux)-1, loc=np.mean(laux), scale=st.sem(laux))
                dicti[list_methods[i]] = {'max': indexH_max, 'min': indexH_min, 'median': indexH_median, 'mean': indexH_mean, 'var': indexH_var, 'std': indexH_std, 'ci': indexH_ci}
                listColor = [list_methods[i],indexH_max, indexH_min, indexH_median, indexH_mean, indexH_var, indexH_std, indexH_ci[0], indexH_ci[1]]
                self.list_color.append(listColor)
                # listdoGrupo.append(dicti)
                # self.data['indexH'] = self.iH
            # lista_fim.append(listdoGrupo)
            dict2[list_methods[c]] = dicti
            self.to_tab(latex_path, 'segreg', dataset, False)
            c += 1
            self.list_color = []


        self.dumpDict(dict2, dict_path, 'expo_iso', dataset)
        self.dumpDict(self.data, dict_path, 'measures', dataset)
        self.to_tab(latex_path, 'segreg', dataset, True)

        # sys.exit(0)

    def to_tab(self, path, file_seg, dataset, dec):

        if dec == True:
            name = path + dataset + '_' + file_seg + '.csv'
            mycsv = csv.writer(open(name, 'w'))
            for row in self.list_csv:
                mycsv.writerow(row)

        else:
            name = path + dataset + '_' + str(self.list_color[0][0][4:]) + '.csv'
            mycsv = csv.writer(open(name, 'w'))
            for row in self.list_color:
                mycsv.writerow(row)


args = sys.argv
# for i, arg in enumerate(sys.argv):
#     print(f"Argument {i:}: {arg}")

path = args[1]
pathin = args[2]
dirfile = args[3]
bw = int(args[4])
sizecell = int(args[5])
method = int(args[6])
# print('path: ', path)
# print('dire: ', dirfile)
# outfile = args[5]
inputpath = pathin + dirfile +'/output/'
outpathshp = path + dirfile + '/shapes/'
outpathmeasures = path + dirfile+ '/results/'
outpathplots = path + dirfile +'/plots/'
measures = int(args[7])
listGroups = [str(args[8]), str(args[9]), str(args[10]), str(args[11])]
inputfiles = args[12:]
# print(outpathshp)
# print(inputpath)
# print(outpathmeasures)
# print(listGroups)
# for i in inputfiles:
#     print(i)
# a = inputpath + inputfiles[0] + '.txt'
# print(a)
# sys.exit(0)
# listGroups = ['AGENT_RED', 'AGENT_YELLO', 'AGENT_BLUE', 'AGENT_CYAN']
#measures are:  expo_local, expo_global, dissimilarity_local, dissimilarity_global
#               entropy_local, entropy_global, indexH_local, indexH_global
# 1 for active, 0 for inactive
# print('input', inputfiles)
u = Utils()
u.createCSV(inputpath, inputpath, inputfiles)
# print('done to csv')
u.createSHP(inputpath, outpathshp, inputfiles)
# print('done to shp')
if measures == 1:
    all_measures = [1, 1, 1, 1, 1, 1, 1, 1]
elif measures == 2:
    all_measures = [0, 1, 0, 1, 0, 0, 0, 0]

segreg = Segreg()
listDict = []
for inst in inputfiles:
    # print(inst)
    segreg.preprocess(outpathshp, inst, listGroups)
    segreg.runMeasures(all_measures, bw, method)
    segreg.saveResults(inst, outpathmeasures)
    dict = u.dataCleaner(outpathmeasures, inst)
    listDict.append(dict)

# print('done to preprocess and measures')
u.pre_plots_full(outpathplots, listDict, inputfiles, (bw/sizecell), outpathmeasures, dirfile)
listFiles = []
lap = ['_segreg', '_red', '_yellow', '_blue', '_cyan']
for i in range(0, 5):
    listFiles.append(dirfile + lap[i])

u.processPlot2(listFiles, outpathmeasures, outpathplots, dirfile)

# sys.exit(0)
