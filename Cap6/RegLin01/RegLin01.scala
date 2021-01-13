import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

def main(): Unit = {
  // Cria uma sessão Spark
  val spark = SparkSession.builder().appName("ExemploRegLin").getOrCreate()

  // Define o nome do arquivo. Se estiver em outro diretório, use file:///
  val path = "dados.txt"

  // Dados de treino
  val training = spark.read.format("libsvm").load(path)
  training.printSchema()

  // Cria um objeto LinearRegression
  val lr = new LinearRegression().setMaxIter(100).setRegParam(0.3)

  // Fit do modelo
  val lrModel = lr.fit(training)

  // Imprime os coeficientes da regressão linear
  println(s"Coeficientes: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  // Sumariza o modelo no dataset de treino e imprime algumas métricas
  val trainingSummary = lrModel.summary
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"R2: ${trainingSummary.r2}")

  // Finaliza a sessão
  spark.stop()
}
main()
