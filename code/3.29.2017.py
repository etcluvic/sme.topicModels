
# coding: utf-8

# In[1]:

#import findspark
#findspark.init()

from collections import defaultdict
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.ml.clustering import LDA, LDAModel
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import udf
from pyspark.sql import SQLContext
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.sql import DataFrameWriter
import time
import re
import pickle
import sys
import codecs

def printSeparator(character, times):
	print(character * times)

def indices_to_terms(vocabulary):
	def indices_to_terms(xs):
		return [vocabulary[int(x)] for x in xs]
	return udf(indices_to_terms, ArrayType(StringType()))


if __name__ == '__main__':

	#encoding
	"""
	print(sys.getdefaultencoding())

	reload(sys)
	sys.setdefaultencoding('utf-8')

	print(sys.getdefaultencoding())
	"""
	#end of encoding



	numStopWordsDiv = 3    # Number of most common words to remove
	numTopics 		= 20
	numIterations 	= 1
	thisSet 		= ''
	savePath 		= 'results/' #use absolute paths!
	entitiesPath 	= 'entities/'
	setNamesPath 	= 'sets/sets.txt'

	commonNames = pickle.load( open( entitiesPath + 'common.names.p', 'rb' ) )
	swEng = pickle.load( open( entitiesPath + 'sw.nltk.eng.p', 'rb' ) )
	swFre = pickle.load( open( entitiesPath + 'sw.nltk.fre.p', 'rb' ) )
	extraStopWords = commonNames + swEng + swFre

	#setNames
	f = open(setNamesPath)
	lines = f.readlines()
	setNames =[l.strip() for l in lines]

	conf = SparkConf()
	#print(sc._conf.getAll())
	conf.set("spark.executor.memory","12g")
	conf.set("spark.driver.memory","12g")
	conf.set("spark.driver.maxResultsSize","0")
	print(conf.getAll())

	for setNumber, thisSet in enumerate(setNames):
		print(setNumber, thisSet)
		inputPath 	= 'fullTextBySets/' + thisSet + '/*.txt'
		iterate 	= True



		#start time
		start_time = time.time()

		sc = SparkContext(conf=conf)

		files = sc.wholeTextFiles(inputPath)
		print('files:', files.count())



		#df = files.toDF(['name', 'content'])
		sqlContext = SQLContext(sc)
		df = sqlContext.createDataFrame(files, ["id", "text"])
		df.persist(StorageLevel.MEMORY_AND_DISK)

		#df.select('id','text').show(5, truncate=False)
		#df.printSchema()
		#df.describe('id').show()

		#udfs - preprocessing
		#F1 = udf(lambda x: len(x.split()), IntegerType())
		step1 		= udf(lambda x: [item.lower() for item in x if len(item)>4], ArrayType(StringType()))
		step2 		= udf(lambda x: [item for item in x if item.isalpha()], ArrayType(StringType()))
		countMe 	= udf(lambda x: len(x), IntegerType())
		parseThetas = udf(lambda x: ', '.join([str(item) for item in x]), StringType())
		parseTerms 	= udf(lambda x: ', '.join([item for item in x]), StringType())
		parseInts 	= udf(lambda x: str(x), StringType())
		#end udfs


		tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
		df = tokenizer.transform(df)

		df = df.withColumn('length', countMe(df["tokens"]))
		#df.select('id','length').show(50, truncate=False)


		df = df.withColumn('filtered', step1(df["tokens"]))
		#df.select('id','Text').show(5)

		df = df.withColumn('filtered2', step2(df["filtered"]))
		#df.select('id','Text').show(5)


		#remove unused columns and rename
		df = df.drop('length', 'filtered','tokens')
		df = df.withColumnRenamed('filtered2', 'tokens')
		#df.printSchema()


		#stop words - begin
		cv = CountVectorizer(inputCol="tokens")
		swModel = cv.fit(df)
		#stop words - end

		#print('sw vocabulary:', len(swModel.vocabulary))
		numStopWords = len(swModel.vocabulary) / numStopWordsDiv
		print('*'*5, numStopWords, '*'*5)

		#remove stop words - for loop should start here


		remover = StopWordsRemover(inputCol="tokens", outputCol="nostop", stopWords=swModel.vocabulary[:numStopWords] + extraStopWords)
		df2 = remover.transform(df)
		df2.persist(StorageLevel.MEMORY_AND_DISK)

		#features
		cv = CountVectorizer(inputCol="nostop", outputCol="features")
		model = cv.fit(df2)
		df2 = model.transform(df2)

		#remove unused column
		df2 = df2.drop('nostop')
		#df2.printSchema()


		#df.show(truncate=False)

		#lda stuff
		lda = LDA(k=numTopics, seed=1, maxIter=numIterations, optimizer="em", topicDistributionCol="theta")
		ldaModel = lda.fit(df2)
		df2 = ldaModel.transform(df2)

		#print('schema:')
		#df.printSchema()
		#df2.select('id','theta').show(5, truncate=False)

		df2 = df2.withColumn('thetaString', parseThetas(df2["theta"]))
		#df2.printSchema()
		#df2.select('id','thetaString').show(5, truncate=False)


		keywords = {}

		topicIndices= ldaModel.describeTopics()
		#vocablist = model.vocabulary

		#topicsRDD = topicIndices.rdd

		ti = topicIndices.withColumn("topics_words", indices_to_terms(model.vocabulary)("termIndices"))
		#ti.persist(StorageLevel.MEMORY_AND_DISK)


		ti = ti.withColumn('topicString', parseInts(ti["topic"]))
		ti = ti.withColumn('termWeightsString', parseThetas(ti["termWeights"]))
		ti = ti.withColumn('topics_wordsString', parseTerms(ti["topics_words"]))



		#print(ti.columns)

		#save file - terms per topic
		#f = open(savePath + 'terms.csv','w')


		#save file - topic distribution and terms

		writer = DataFrameWriter(df2.select('id','thetaString'))
		writer.csv('file://' + savePath + thisSet + '.thetas')

		writer = DataFrameWriter(ti.select('topicString','termWeightsString','topics_wordsString'))
		writer.csv('file://' + savePath + thisSet + '.terms')


		"""
		#save file - terms per topic
		f = open(savePath + thisSet + '.terms.csv','w')

		for row in ti2:
			items = len(row[3])
			for i in range(items):
				f.write(str(row[0]) + ',' + row[3][i].encode('utf-8') + ',' + str(row[2][i]) + '\n')

		f.close()
		"""
		elapsed_time = time.time() - start_time
		print('elapsed time in seconds:')
		print(elapsed_time)

		#save file - metadata
		f = open(savePath + thisSet +'.meta.txt','w')
		f.write('time: '+ str(elapsed_time) +'\n')
		f.write('stopWordsRemoved: '+ str(numStopWords) + '\n')
		f.write('topics: ' + str(numTopics) + '\n')
		f.write('iterations: ' + str(numIterations) + '\n')
		f.close()

		print('stopping sc...')
		sqlContext.clearCache()
		sc.stop()
		del sc


		printSeparator('*', 50)



	print('bye...')



# In[ ]:




# In[ ]:
