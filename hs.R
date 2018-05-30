if("pROC" %in% rownames(installed.packages()) == FALSE) {
  install.packages("pROC")
}
require(pROC)

source('globalSearch.R')

AUC <- function(class1, class2){
  result = tryCatch({
    
  }, warning = function(w) {
    0
  }, error = function(e) {
    0
  })
  return (result)
}


initializeVars <- function(dataFile, lableCol = 1, hSize = 30, dataCat="Cancer"){
  rawData <<- read.csv(dataFile)
  
  if(dataCat == "seizure") rawData <<- rawData[!duplicated(rawData$V),]
  
  globalDataset <<- rawData[-lableCol]
  globalLabel <<- rawData[lableCol]
  globalLabel <<- as.numeric(t(globalLabel))
  
  inputSize <<- length(globalDataset)
  outSize <<- ceiling(log2(length(table(globalLabel))))
  
  if(dataCat == "seizure") inputSize <<- 10
  
  hiddenSize <<- hSize
}

folds <- function (optimizer = harmonySearch, nural="lstm"){
  wieghtsSize = 0
  if(nural == "lstm"){
    wieghtsSize = inputSize * hiddenSize * 4 + hiddenSize * hiddenSize * 4 + hiddenSize * outSize
  }
  else{
    wieghtsSize = inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize * outSize
  }
    
  
  testNo = floor(length(globalDataset[,1]) /5)
  testSample = 1:testNo
  results <<- c()
  
  for (i in 0:5){
    dataset <<- globalDataset[-(testSample + testNo * i),]
    label <<- globalLabel[-(testSample + testNo * i)]
    
    optWieghts = optimizer(wieghtsSize, nural)
    
    dataset <<- globalDataset[(testSample + testNo * i),]
    label <<- globalLabel[(testSample + testNo * i)]
    pridict <<- classify(optWieghts, nural)
    results[1 + i] <<- sum(label== pridict)/length(label)
    
  }
  return(sum(results) / 5)
}


main <- function(){
  running <<- FALSE
  foldResults <<- c()
  file <- "seizure.csv" #file.choose()
  
  initializeVars(file, lableCol = 3, dataCat="seizure")
  
  for(i in 1:10){
    iterations <<- 5 * i
    foldResults[i] <<- folds()
  }
  plot(1:10, foldResults)
  running <<- TRUE
}

main()

