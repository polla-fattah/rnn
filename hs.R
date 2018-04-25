require(pROC)
source('globalSearch.R')

AUC <- function(class1, class2) multiclass.roc(class1, class2)$auc[1]



initializeVars <- function(dataFile, lableCol = 1, hSize = 10){
  rawData <<- read.csv(dataFile)
  
  dataset <<- rawData[-lableCol]
  label <<- rawData[lableCol]
  label <<- as.numeric(t(label))
  
  inputSize <<- length(dataset)
  outSize <<- ceiling(log2(length(table(label))))
  
  hiddenSize <<- hSize
  
}

main <- function(){
  file <- "iris.csv" #file.choose()
  initializeVars(file, lableCol = 5)
  wieghtsSize = inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize * outSize
  #print(wieghtsSize)
  #defEvo(wieghtsSize)
  #harmonySearch(wieghtsSize)
  prticleSwarmOpt(wieghtsSize)
  #antLoinOpt(wieghtsSize)
  
}

main()