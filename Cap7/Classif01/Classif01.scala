// Regressão Logística

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

object LogisticRegressionExample {

  def main(): Unit = {
    val spark = SparkSession
      .builder
      .appName("LogisticRegressionExample")
      .getOrCreate()

    // Carregando os dados de treino
    val training = spark.read.format("libsvm").load("sample_libsvm_data.txt")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit do modelo
    val lrModel = lr.fit(training)

    // Intercepto e coeficientes da regressão logística
    println(s"Coeficientes: ${lrModel.coefficients} Intercepto: ${lrModel.intercept}")

    spark.stop()
  }
}
