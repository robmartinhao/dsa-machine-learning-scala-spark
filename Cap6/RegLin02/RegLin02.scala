// Regressão Linear com Scala e Spark

// Módulos
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

// Definindo o nível do log
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Inicializando a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Preparando os dados de treino e de teste
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("casas.csv")

// Verificando os dados
data.printSchema()

// Imprimindo uma linha do dataset
val colnames = data.columns
val firstrow = data.head(1)(0)

println("\n")
println("Linha do dataset")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

// Configurando um Dataframe 


// Algumas coisas que precisamos fazer antes que o Spark possa aceitar os dados!
// Definir o dataset na forma de duas colunas ("label", "features")

// Isso nos permitirá juntar várias colunas de recursos em uma única coluna de uma matriz de valores feautre
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Criando o dataframe
val df = data.select(data("Price").as("label"),$"Avg Area Income",$"Avg Area House Age",$"Avg Area Number of Rooms",$"Area Population")

// Um assembler converte os valores de entrada em um vetor
// Um vetor é o que o algoritmo ML lê para treinar um modelo

// Define as colunas de entrada das quais devemos ler os valores
// Define o nome da coluna onde o vetor será armazenado
val assembler = new VectorAssembler().setInputCols(Array("Avg Area Income","Avg Area House Age","Avg Area Number of Rooms","Area Population")).setOutputCol("features")

// Transformamos o dataset em um objeto de duas colunas, no formato esperado pelo modelo
val output = assembler.transform(df).select($"label",$"features")

//Imprimindo a versão final do dataframe que vai alimentar o modelo de regressão
output.show()


// Configurando o modelo de regressão


// Criar um objeto de Regressão Linear
val lr = new LinearRegression()

// Grid de Parâmetros
// val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).addGrid(lr.fitIntercept).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).build()

// Treino e teste
// val trainValidationSplit = new TrainValidationSplit().setEstimator(lr).setEvaluator(new RegressionEvaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)

// Fit do modelo nos dados

// Nota: Mais tarde veremos por que devemos dividir os dados em primeiro lugar, mas por agora vamos ajustar a todos os dados.
val lrModel = lr.fit(output)

// Imprimir os coeficientes e interceptar para regressão linear
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


// Avaliação

// Resumindo o Modelo
val trainingSummary = lrModel.summary

// Parâmetros
println(s"numIterations: ${trainingSummary.totalIterations}")

// Resíduos e Previsões
trainingSummary.residuals.show()
trainingSummary.predictions.show()

// Métricas
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"R2: ${trainingSummary.r2}")




