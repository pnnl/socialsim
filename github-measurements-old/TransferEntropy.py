import pandas as pd
import numpy as np
import jpype
from jpype import *
from datetime import datetime

'''
Notice: This computer software was prepared by Battelle Memorial Institute, hereinafter the Contractor, under Contract
No. DE-AC05-76RL01830 with the Department of Energy (DOE).  All rights in the computer software are reserved by DOE on
behalf of the United States Government and the Contractor as provided in the Contract.  You are authorized to use this
computer software for Governmental purposes but it is not to be released or distributed to the public.  NEITHER THE
GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
SOFTWARE.  This notice including this sentence must appear on any copies of this computer software.
'''

'''
This class implements measurements to calculate the transfer entropy between users. The main function
for TE calculation requires the jpype package. This is a java package that has to be called
within python by using a Java Virtual Machine.

These measurements assume that the data is in the order id,created_at,type,actor.id,repo.id
'''

'''
This method takes a list of times and transforms them to a time series

Input: List of created times
Output: List representing a time series (differences)
'''
def getTimeSeriesInSecs(ts_list):
    base_time = datetime.strptime('2015-01-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
    secSet = set()
    for timeVal in ts_list:
        time_std = datetime.strptime(timeVal, '%Y-%m-%dT%H:%M:%SZ')
        diff = time_std - base_time
        secSet.add(int(diff.total_seconds()))

    secList = sorted(list(secSet))
    return secList

'''
This method bins the time series into discrete bins

Input: totalBins - Total Number of bins
       binSize - Size of the bins
       timeSeries - This is the list representation of the time series
'''
def getBinnedTimeSeriesSingleBinary(totalBins, binSize, timeSeries):
    tsBinned = np.zeros((totalBins), dtype=int)
    for timeVal in timeSeries:
        idx = (timeVal // binSize)
        tsBinned[idx] = 1

    return tsBinned

'''
This method bins the time series into real valued bins

Input: totalBins - Total Number of Bins
       binSize - Size of the bins
       timeSeries - This is the list representation of the time series
'''
def getBinnedTimeSeriesSingleRealVal(totalBins, binSize, timeSeries):
    tsBinned = np.zeros((totalBins), dtype=float)
    for timeVal in timeSeries:
        idx = int((timeVal // binSize))
        tsBinned[idx] = tsBinned[idx] + 1.00

    return tsBinned.tolist()

'''
This method calculates the transfer entropy (TE) between two binary time series

Input: src - This is the source time series
       dest - This is the destination time series
       delayParam - This is the parameter that controls the delay when calculating the TE

Output: Value of Transfer Entropy between the source and destination time series.
'''
def getTETimeSeriesPairBinary(src, dest, delayParam):
    teCalcClass = jpype.JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
    teCalc = teCalcClass(2, 1, 1, 1, 1, delayParam)

    teCalc.initialise()
    teCalc.addObservations(src, dest)
    te = teCalc.computeAverageLocalOfObservations()

    return te

'''
This method calculates the transfer entropy (TE) between two real time series

Input: src - This is the source time series
       dest - This is the destination time series
       delayParam - This is the parameter that controls the delay when calculating the TE

Output: Value of Transfer Entropy between the source and destination time series.
'''
def getTETimeSeriesPairRealValued(src, dest, delay):
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true")  # Normalise the individual variables
    teCalc.setProperty("k", "3")  # Use Kraskov parameter K=4 for 4 nearest points

    teCalc.initialise(1, 1, 1, 1, delay)  # Use history length 1 (Schreiber k=1)
    teCalc.setObservations(JArray(JDouble, 1)(src), JArray(JDouble, 1)(dest))
    te = teCalc.computeAverageLocalOfObservations()

    return te


'''
This method calculates the Transfer entropy for two users and a given dataframe

Input: df - Data frame to extract user data from. This can be any subset of data
       user1 - The id of the first user (source user)
       user2 - The id of the second user (destination user)
       realSeries - Boolean that indicates whether or not the time series should be binned into real or discrete values

Output: Transfer Entropy between the two users
'''
def getTransferEntropy(df,user1,user2,realSeries=False):

    df.columns = ['id', 'time', 'type', 'user', 'repo']

    user1Series = df[df.user == user1]['time'].tolist()
    user2Series = df[df.user == user2]['time'].tolist()
    user1Series = getTimeSeriesInSecs(user1Series)
    user2Series = getTimeSeriesInSecs(user2Series)

    binSize = 10800  # 3 hours = 10800 secs
    maxTime = max(max(user1Series), max(user2Series))
    totalbins = int(np.ceil(maxTime / float(binSize)))

    te = 0.0

    ##Jar location for the infodynamics package
    jarLocation = "./infodynamics.jar"

    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


    if realSeries:
        user1Series = getBinnedTimeSeriesSingleRealVal(totalbins,binSize,user1Series)
        user2Series = getBinnedTimeSeriesSingleRealVal(totalbins,binSize,user2Series)
        te = getTETimeSeriesPairRealValued(user1Series, user2Series, 3)
    else:
        user1Series = getBinnedTimeSeriesSingleBinary(totalbins, binSize, user1Series)
        user2Series = getBinnedTimeSeriesSingleBinary(totalbins, binSize, user2Series)
        te = getTETimeSeriesPairBinary(user1Series,user2Series,1)

    jpype.shutdownJVM()

    return te





