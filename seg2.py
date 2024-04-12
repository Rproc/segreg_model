from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
import numpy as np
import shapefile
import geopandas as gpd
import sys

import os.path
from scipy.spatial.distance import cdist

class Segreg(object):
    def __init__(self):
        """Constructor.
        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """

        self.measuresEmpty = True

        # Segregation measures attributes
        self.attributeMatrix = np.matrix([])    # attributes matrix full size - all columns
        self.location = []                      # x and y coordinates from tract centroid (2D lists)
        self.pop = []                           # population of each groups by tract (2D lists)
        self.pop_sum = []                       # total population of the tract (sum all groups)
        self.locality = []                      # population intensity by groups by tract
        self.n_location = 0                     # length of list (n lines) (attributeMatrix.shape[0])
        self.n_group = 0                        # number of groups (attributeMatrix.shape[1] - 4)
        self.costMatrix = []                    # scipy cdist distance matrix
        self.tract_id = []                      # tract ids in string format

        # Local and global internals
        self.local_dissimilarity = []
        self.local_exposure = []
        self.local_entropy = []
        self.local_indexh = []
        self.global_dissimilarity = []
        self.global_exposure = []
        self.global_entropy = []
        self.global_indexh = []

        self.diss_local = False
        self.expo_local = False
        self.entro_local = False
        self.idxh_local = False

    def clearVariables(self):
        """clear local lists and variables"""
        # clear input tables
        self.location = []
        self.pop = []
        self.pop_sum = []
        self.locality = []
        self.n_location = 0
        self.n_group = 0
        self.tract_id = []
        self.selectedFields = []
        self.layers = []

        # clear result tables
        self.local_dissimilarity = []
        self.local_exposure = []
        self.local_entropy = []
        self.local_indexh = []
        self.global_dissimilarity = []
        self.global_exposure = []
        self.global_entropy = []
        self.global_indexh = []

    def preprocess(self, inputpath, filename, listGroup):
        # print(inputpath)
        # print(filename)
        # print(listGroup)
        pathfile = inputpath + filename + '.shp'
        poly = gpd.read_file(pathfile)
        # print(poly.columns)
        points = poly.copy()

        points.geometry = points['geometry'].centroid
        points.crs = poly.crs
        centroidX, centroidY = points['geometry'][:].x, points['geometry'][:].y

        centroidX = np.asarray(centroidX)
        centroidX = np.reshape(centroidX, (len(centroidX), 1))
        centroidY = np.asarray(centroidY)
        centroidY = np.reshape(centroidY, (len(centroidY), 1))

        id_values = [str(x) for x in points['AGENT_ID']]
        self.tract_id = np.asarray(id_values)
        self.tract_id = self.tract_id.reshape((len(id_values), 1))
        groups = []
        # print(listGroup)
        # sys.exit(0)

        for gp in listGroup:
            groups.append(list(map(int, points[gp][:])))

        groups = np.asarray(groups).T

        data = np.concatenate((centroidX, centroidY, groups), axis=1)
        self.attributeMatrix = np.asmatrix(data)
        n = self.attributeMatrix.shape[1]
        self.location = self.attributeMatrix[:, 0:2]
        self.location = self.location.astype('float')
        self.pop = self.attributeMatrix[:, 2:n]
        self.pop[np.where(self.pop < 0)[0], np.where(self.pop < 0)[1]] = 0.0
        self.n_group = n - 2
        self.n_location = self.attributeMatrix.shape[0]
        self.pop_sum = np.sum(self.pop, axis=1)

    def getWeight(self, distance, bandwidth, weightmethod=1):
        """
        Compute the weights for neighborhood.
        :param distance: distance in meters to be considered for weighting
        :param bandwidth: bandwidth in meters selected to perform neighborhood
        :param weightmethod: method to be used: 1-gussian , 2-bi square and 3-moving window
        :return: weight value for internal use
        """
        distance = np.asarray(distance.T)

        if weightmethod == 1:
            weight = np.exp((-0.5) * (distance / bandwidth) * (distance / bandwidth))

        elif weightmethod == 2:
            weight = (1 - (distance / bandwidth) * (distance / bandwidth)) * (
            1 - (distance / bandwidth) * (distance / bandwidth))
            sel = np.where(distance > bandwidth)
            weight[sel[0]] = 0

        elif weightmethod == 3:
            weight = (1 + (distance * 0))
            sel = np.where(distance > bandwidth)
            weight[sel[0]] = 0

        else:
            raise Exception('Invalid weight method selected!')

        return weight

    def cal_localityMatrix(self, bandwidth, weightmethod):
        """
        Compute the local population intensity for all groups.
        :param bandwidth: bandwidth for neighborhood in meters
        :param weightmethod: 1 for gaussian, 2 for bi-square and empty for moving window
        :return: 2d array like with population intensity for all groups
        """
        n_local = self.location.shape[0]
        n_subgroup = self.pop.shape[1]
        locality_temp = np.empty([n_local, n_subgroup])

        for index in range(0, n_local):
            for index_sub in range(0, n_subgroup):
                cost = cdist(self.location[index, :], self.location)
                weight = self.getWeight(cost, bandwidth, weightmethod)
                locality_temp[index, index_sub] = np.sum(weight * np.asarray(self.pop[:, index_sub]))/np.sum(weight)

        self.locality = locality_temp
        # assign zero to negative values
        self.locality[np.where(self.locality < 0)[0], np.where(self.locality < 0)[1]] = 0

    def cal_localDissimilarity(self):
        """
        Compute local dissimilarity for all groups.
        """
        # non-spatial version loop, uses raw data
        if len(self.locality) == 0:
            lj = np.ravel(self.pop_sum)
            tjm = np.asarray(self.pop) * 1.0 / lj[:, None]
            tm = np.sum(self.pop, axis=0) * 1.0 / np.sum(self.pop)
            index_i = np.sum(np.asarray(tm) * np.asarray(1 - tm))
            pop_total = np.sum(self.pop)
            local_diss = np.sum(1.0 * np.array(np.fabs(tjm - tm)) *
                                np.asarray(self.pop_sum).ravel()[:, None] / (2 * pop_total * index_i), axis=1)

        # spatial version loop, uses population intensity
        else:
            lj = np.asarray(np.sum(self.locality, axis=1))
            tjm = self.locality * 1.0 / lj[:, None]
            tm = np.sum(self.pop, axis=0) * 1.0 / np.sum(self.pop)
            index_i = np.sum(np.asarray(tm) * np.asarray(1 - tm))
            pop_total = np.sum(self.pop)
            local_diss = np.sum(1.0 * np.array(np.fabs(tjm - tm)) *
                                np.asarray(self.pop_sum).ravel()[:, None] / (2 * pop_total * index_i), axis=1)

        # clear nan values and transpose matrix
        local_diss = np.nan_to_num(local_diss)
        local_diss = np.asmatrix(local_diss).transpose()
        self.local_dissimilarity = local_diss

    def cal_globalDissimilarity(self):
        """
        Compute global dissimilarity calling the local version and summing up.
        """
        local_diss = self.local_dissimilarity
        self.global_dissimilarity = np.sum(local_diss)

    def cal_localExposure(self):
        """
        Compute the local exposure index of group m to group n.
        in situations where m=n, then the result is the isolation index.
        """
        m = self.n_group
        j = self.n_location
        exposure_rs = np.zeros((j, (m * m)))

        # non-spatial version loop, uses raw data
        if len(self.locality) == 0:
            local_expo = np.asarray(self.pop) * 1.0 / np.asarray(np.sum(self.pop, axis=0)).ravel()
            locality_rate = np.asarray(self.pop) * 1.0 / np.asarray(np.sum(self.pop, axis=1)).ravel()[:, None]
            for i in range(m):
                exposure_rs[:, ((i * m) + 0):((i * m) + m)] = np.asarray(locality_rate) * \
                                                              np.asarray(local_expo[:, i]).ravel()[:, None]
        # spatial version loop, uses population intensity
        else:
            local_expo = np.asarray(self.pop) * 1.0 / np.asarray(np.sum(self.pop, axis=0)).ravel()
            locality_rate = np.asarray(self.locality) * 1.0 / np.asarray(np.sum(self.locality, axis=1)).ravel()[:, None]
            for i in range(m):
                exposure_rs[:, ((i * m) + 0):((i * m) + m)] = np.asarray(locality_rate) * \
                                                              np.asarray(local_expo[:, i]).ravel()[:, None]

        # clear nan and inf values and convert to matrix
        exposure_rs[np.isinf(exposure_rs)] = 0
        exposure_rs[np.isnan(exposure_rs)] = 0
        exposure_rs = np.asmatrix(exposure_rs)
        self.local_exposure = exposure_rs

    def cal_globalExposure(self):
        """
        Compute global exposure calling the local version and summing up.
        """
        m = self.n_group
        local_exp = self.local_exposure
        global_exp = np.sum(local_exp, axis=0)
        global_exp = global_exp.reshape((m, m))
        self.global_exposure = global_exp

    def cal_localEntropy(self):
        """
        Compute local entropy score for a unit area Ei (diversity). A unit
        within the metropolitan area, such as a census tract. If population
        intensity was previously computed, the spatial version will be returned,
        otherwise the non spatial version will be selected (raw data).
        """
        # non-spatial version, uses raw data
        if len(self.locality) == 0:
            proportion = np.asarray(self.pop / self.pop_sum)

        # spatial version, uses population intensity
        else:
            polygon_sum = np.sum(self.locality, axis=1).reshape(self.n_location, 1)
            proportion = np.asarray(self.locality / polygon_sum)

        entropy = proportion * np.log(1 / proportion)

        # clear nan and inf values, sum line and reshape
        entropy[np.isnan(entropy)] = 0
        entropy[np.isinf(entropy)] = 0
        entropy = np.sum(entropy, axis=1)
        entropy = entropy.reshape((self.n_location, 1))
        self.local_entropy = entropy

    def cal_globalEntropy(self):
        """
        Compute the global entropy score E (diversity), metropolitan area's entropy score.
        """
        group_score = []
        pop_total = np.sum(self.pop_sum)
        prop = np.asarray(np.sum(self.pop, axis=0))[0]

        # loop at sum of each population groups
        for group in prop:
            group_idx = group / pop_total * np.log(1 / (group / pop_total))
            group_score.append(group_idx)

        # sum scores from each group to get the result
        global_entro = np.sum(group_score)
        self.global_entropy = global_entro

    def cal_localIndexH(self):
        """
        Computes the local entropy index H for all localities. The functions
        cal_localEntropy() for local diversity and cal_globalEntropy for global
        entropy are called as input. If population intensity was previously
        computed, the spatial version will be returned, else the non spatial
        version will be selected (raw data).
        """
        local_entropy = self.local_entropy
        global_entropy = self.global_entropy

        # compute index
        et = np.asarray(global_entropy * np.sum(self.pop_sum))
        eei = np.asarray(global_entropy - local_entropy)
        h_local = np.asarray(self.pop_sum) * eei / et

        self.local_indexh = h_local

    def cal_globalIndexH(self):
        """
        Compute global index H calling the local version summing up.
        """
        h_local = self.local_indexh
        h_global = np.sum(h_local)
        self.global_indexh = h_global

    def runMeasures(self, all_measures, bw, metric):
        """
        Call the functions to compute local and global measures. The dependency
        complexity is handle by chacking the flaged measures and calling local
        measures for global versions. Results are stored for posterior output save.
        """

        # check if there is at least one measure selected
        if not all_measures or all(a == 0 for a in all_measures):
            # print('selecione alguma metrica')
            return
        else:
            self.cal_localityMatrix(bw, metric)
            # expo_local, expo_global, dissimilarity_local, dissimilarity_global
            # entropy_local, entropy_global, indexH_local, indexH_global
            # call local and global exposure/isolation measures
            if all_measures[1] == True:
                self.cal_localExposure()
                self.cal_globalExposure()
                self.expo_local = True
            if all_measures[0] == True and len(self.local_exposure) == 0:
                self.cal_localExposure()
                self.expo_local = True

            # call local and global dissimilarity measures
            if all_measures[3] == True:
                self.cal_localDissimilarity()
                self.cal_globalDissimilarity()
                self.diss_local = True
            if all_measures[2] == True and len(self.local_dissimilarity) == 0:
                self.cal_localDissimilarity()
                self.diss_local = True

            # call local and global entropy measures
            if all_measures[5] == True:
                self.cal_localEntropy()
                self.cal_globalEntropy()
                self.entro_local = True
            if all_measures[4] == True and len(self.local_entropy) == 0:
                self.cal_localEntropy()
                self.entro_local = True

            # call local and global index H measures
            if all_measures[7] == True:
                self.cal_localEntropy()
                self.cal_globalEntropy()
                self.cal_localIndexH()
                self.cal_globalIndexH()
                self.idxh_local = True
            if all_measures[6] == True and len(self.local_indexh) == 0:
                self.cal_localEntropy()
                self.cal_globalEntropy()
                self.cal_localIndexH()
                self.idxh_local = True
            # inform sucess if all were computed
            # QMessageBox.information(None, "Info", 'Measures computed successfully!')
            self.measuresEmpty = False

    def joinResultsData(self):
        """ Join results on a unique matrix and assign names for columns to be
        used as header for csv file and shapefile output"""
        names = ['id','x','y']
        measures_computed = []

        # create new names for groups starting by 0
        for i in range(self.n_group):
            names.append('group_' + str(i))

        # update names with locality if computed
        if len(self.locality) != 0:
            measures_computed.append(self.locality)
            for i in range(self.n_group):
                names.append('intens_' + str(i))

        # update names with exposure/isolation if computed
        if self.expo_local == True:
            measures_computed.append(self.local_exposure)
            for i in range(self.n_group):
                for j in range(self.n_group):
                    if i == j:
                        names.append('iso_' + str(i) + str(j))
                    else:
                        names.append('exp_' + str(i) + str(j))

        # update names with dissimilarity if computed
        if self.diss_local == True:
            measures_computed.append(self.local_dissimilarity)
            names.append('dissimil')

        # update names with entropy if computed
        if self.entro_local == True:
            measures_computed.append(self.local_entropy)
            names.append('entropy')

        # update names with index H if computed
        if self.idxh_local == True:
            measures_computed.append(self.local_indexh)
            names.append('indexh')

        # output_labels = tuple([eval(x) for x in measures_computed])
        output_labels = measures_computed
        # try to concaneta results, else only original input
        try:
            computed_results = np.concatenate(output_labels, axis=1)
            results_matrix = np.concatenate((self.tract_id, self.attributeMatrix, computed_results), axis=1)
            measures_computed[:] = []
            return results_matrix, names
        except ValueError:
            results_matrix = np.concatenate((self.tract_id, self.attributeMatrix), axis=1)
            return results_matrix, names

    def saveResults(self, filename, path):
        """ Save results to a local file."""

        # filename, __ = QFileDialog.getSaveFileName(self.dlg, "Select output file ", "", "*.csv")
        # self.dlg.leOutput.setText(filename)
        # path = self.dlg.leOutput.text()
        result = self.joinResultsData()
        labels = str(', '.join(result[1]))

        savefile = path + filename + '.csv'
        # save local measures results on a csv file
        np.savetxt(savefile, result[0], header=labels, delimiter=',', newline='\n', fmt="%s")

        # add result to canvas as shapefile if requested

        savefile = path + filename + '_global.csv'
        # save global results to a second csv file
        with open(savefile, "w") as f:
            f.write('Global dissimilarity: ' + str(self.global_dissimilarity))
            f.write('\nGlobal entropy: ' + str(self.global_entropy))
            f.write('\nGlobal Index H: ' + str(self.global_indexh))
            f.write('\nGlobal isolation/exposure: \n')
            f.write(str(self.global_exposure))

        # clear local variables after save
        self.local_dissimilarity = []
        self.local_exposure = []
        self.local_entropy = []
        self.local_indexh = []


# args = sys.argv
# # for i, arg in enumerate(sys.argv):
# #     print(f"Argument {i:}: {arg}")
#
# inputpath = args[1]
# inputfile = args[2]
# bw = int(args[3])
# method = int(args[4])
# outfile = args[5]
# outpath = args[6]
# measures = int(args[7])
# listGroups = [str(args[8]), str(args[9]), str(args[10]), str(args[11])]
# # listGroups = ['AGENT_RED', 'AGENT_YELLO', 'AGENT_BLUE', 'AGENT_CYAN']
# #measures are:  expo_local, expo_global, dissimilarity_local, dissimilarity_global
# #               entropy_local, entropy_global, indexH_local, indexH_global
# # 1 for active, 0 for inactive
# # print(type(bw), type(method))
# # sys.exit()
#
# if measures == 1:
#     all_measures = [1, 1, 1, 1, 1, 1, 1, 1]
# elif measures == 2:
#     all_measures = [0, 1, 0, 1, 0, 0, 0, 0]
#
# segreg = Segreg()
# segreg.preprocess(inputpath, inputfile, listGroups)
# # segreg.cal_localityMatrix(250, 1)
# segreg.runMeasures(all_measures, bw, method)
# segreg.saveResults(outfile, outpath)
# # segreg.cal_localDissimilarity()
# # segreg.cal_globalDissimilarity()
# print(segreg.global_dissimilarity, segreg.global_entropy, segreg.global_indexh)








# -----------------------
