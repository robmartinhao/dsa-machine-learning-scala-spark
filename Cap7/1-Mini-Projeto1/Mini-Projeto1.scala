// Mini-Projeto 1

// Neste projeto, estaremos trabalhando com um conjunto de dados (fake) de publicidade, indicando se um usuário de internet específico clicou ou não em uma propaganda. 
// Vamos tentar criar um modelo que preveja se clicará ou não em um anúncio baseado nos recursos desse usuário.  

// Este conjunto de dados contém os seguintes recursos:
//    'Daily Time Spent on Site': Tempo diário navegando pelo site
//    'Age': Idade do cliente em anos
//    'Area Income': Média da Renda na área geográfica do consumidor
//    'Daily Internet Usage': Média de uso da internet por dia
//    'Ad Topic Line': Título do anúncio
//    'City': Cidade do consumidor
//    'Male': Se o consumidor era ou não homem
//    'Country': País 
//    'Timestamp': Hora em que o consumidor clicou no anúncio ou na janela fechada
//    'Clicked on Ad': 0 ou 1 indicando clique ou não no anúncio


// Import dos módulos SparkSession e Logisitic Regression
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

// Definindo o nível de informação no log (nesse caso, ERRO)
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Spark Session
val spark = SparkSession.builder().getOrCreate()

// Carregando os dados
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("advertising.csv")

// Print do schema do dataset
data.printSchema()


// Visualizando os dados
// Imprima uma linha de amostra dos dados (várias maneiras de fazer isso)
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Linha de exemplo no dataset")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}


// Data Wrangling

// Façamos o seguinte:
// - Mude o nome da coluna Clicked on Ad para "label"
// - Pegue as seguintes colunas "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"
// - Crie uma nova coluna chamada "Hour", extraída do Timestamp, contendo a Hora do clique

val timedata = data.withColumn("Hour",hour(data("Timestamp")))

val logregdata = (timedata.select(data("Clicked on Ad").as("label"),
                    $"Daily Time Spent on Site", $"Age", $"Area Income",
                    $"Daily Internet Usage",$"Hour",$"Male"))


// Import dos módulos VectorAssembler e Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Cria um novo objeto VectorAssembler chamado assembler as features
// Defina a coluna de saída
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income",
                  "Daily Internet Usage","Hour"))
                  .setOutputCol("features") )


// Use randomSplit para criar uma divisão em treino e teste em 70/30
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)


// Definindo o Pipeline

// Import do Pipeline
import org.apache.spark.ml.Pipeline

// Crie um novo objeto LogisticRegression chamado lr
val lr = new LogisticRegression()

// Crie um novo pipeline com as etapas: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler, lr))

// Ajustar o pipeline ao conjunto de treinamento
val model = pipeline.fit(training)

// Obter resultados no conjunto de teste com transformação
val results = model.transform(test)

// Avaliação do Modelo

// Para métricas e avaliação, importe MulticlassMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Converta os resultados do teste em um RDD usando .as e .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instanciar um novo objeto MulticlassMetrics
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
println("Acurácia:")
println(metrics.accuracy)


